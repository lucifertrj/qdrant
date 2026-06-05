use std::path::Path;

use common::bitvec::{BitSlice, BitVec};
use common::counter::hardware_counter::HardwareCounterCell;
use common::generic_consts::AccessPattern;
use common::mmap::AdviceSetting;
use common::types::PointOffsetType;
use common::universal_io::UniversalRead;

use crate::common::flags::dynamic_stored_flags::DynamicStoredFlags;
use crate::common::live_reload::LiveReload;
use crate::common::operation_error::OperationResult;
use crate::data_types::named_vectors::CowVector;
use crate::data_types::primitive::PrimitiveVectorElement;
use crate::types::{Distance, VectorStorageDatatype};
use crate::vector_storage::VectorStorageRead;
use crate::vector_storage::chunked_vectors::ChunkedVectorsRead;
use crate::vector_storage::multi_dense::appendable_mmap_multi_dense_vector_storage::{
    DELETED_DIR_PATH, MultivectorMmapOffset, OFFSETS_DIR_PATH, VECTORS_DIR_PATH,
    flattened_to_multi_vector, read_multi_vector,
};

#[derive(Debug)]
pub struct ReadOnlyChunkedMultiDenseVectorStorage<T: PrimitiveVectorElement, S: UniversalRead> {
    vectors: ChunkedVectorsRead<T, S>,
    offsets: ChunkedVectorsRead<MultivectorMmapOffset, S>,
    /// Flags marking deleted vectors
    ///
    /// Structure grows dynamically, but may be smaller than actual number of vectors. Must not
    /// depend on its length.
    deleted: BitVec,
    distance: Distance,
    deleted_count: usize,
    /// Chunk-open settings retained so [`LiveReload`] can refresh the chunked
    /// vectors and offsets with the same residency.
    advice: AdviceSetting,
    populate: bool,
}

impl<T: PrimitiveVectorElement, S: UniversalRead> ReadOnlyChunkedMultiDenseVectorStorage<T, S> {
    /// Open the read-only counterpart of
    /// [`AppendableMmapMultiDenseVectorStorage`][1] at `path`, threading every
    /// file open through `fs`.
    ///
    /// Reads the same on-disk layout the writable storage maintains — the
    /// chunked `vectors/` and `offsets/` directories and the `deleted/` flags —
    /// but never creates or writes anything. The deleted flags are materialized
    /// into an owned bitvec via [`DynamicStoredFlags::load_bitvec`]. `populate`
    /// warms the vector and offset chunks (mirroring `is_on_disk = !populate`);
    /// it does not apply to the always-resident deleted bitvec.
    ///
    /// [1]: super::super::appendable_mmap_multi_dense_vector_storage::AppendableMmapMultiDenseVectorStorage
    #[allow(dead_code)] // pending: read-only vector storage enum will use this
    pub fn open(
        fs: &S::Fs,
        path: &Path,
        dim: usize,
        distance: Distance,
        advice: AdviceSetting,
        populate: bool,
    ) -> OperationResult<Self> {
        let vectors = ChunkedVectorsRead::open(
            fs,
            &path.join(VECTORS_DIR_PATH),
            dim,
            advice,
            Some(populate),
        )?;

        // Offsets store one `MultivectorMmapOffset` element per point, so the
        // chunked storage dimensionality is 1.
        let offsets =
            ChunkedVectorsRead::open(fs, &path.join(OFFSETS_DIR_PATH), 1, advice, Some(populate))?;

        let deleted = DynamicStoredFlags::<S>::load_bitvec(fs, &path.join(DELETED_DIR_PATH))?;
        let deleted_count = deleted.count_ones();

        Ok(Self {
            vectors,
            offsets,
            deleted,
            distance,
            deleted_count,
            advice,
            populate,
        })
    }
}

impl<T: PrimitiveVectorElement, S: UniversalRead> LiveReload
    for ReadOnlyChunkedMultiDenseVectorStorage<T, S>
{
    type Fs = S::Fs;

    /// Refresh from disk: reload both the chunked vectors and offsets (picking
    /// up appended multivectors and new chunks), then apply the authoritative
    /// `deleted_points` to the in-memory deletion bitvec. Newly added points are
    /// served straight from the refreshed chunks, so `new_points` is unused.
    fn live_reload(
        &mut self,
        fs: &S::Fs,
        deleted_points: &[PointOffsetType],
        _new_points: &[PointOffsetType],
        _hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        self.vectors.live_reload(fs, self.advice, self.populate)?;
        self.offsets.live_reload(fs, self.advice, self.populate)?;

        for &point in deleted_points {
            let index = point as usize;
            if index >= self.deleted.len() {
                self.deleted.resize(index + 1, false);
            }
            if !self.deleted.replace(index, true) {
                self.deleted_count += 1;
            }
        }

        Ok(())
    }
}

impl<T: PrimitiveVectorElement, S: UniversalRead> VectorStorageRead
    for ReadOnlyChunkedMultiDenseVectorStorage<T, S>
{
    fn distance(&self) -> Distance {
        self.distance
    }

    fn datatype(&self) -> VectorStorageDatatype {
        T::datatype()
    }

    fn is_on_disk(&self) -> bool {
        self.vectors.is_on_disk()
    }

    fn total_vector_count(&self) -> usize {
        self.offsets.len()
    }

    fn get_vector<P: AccessPattern>(&self, key: PointOffsetType) -> CowVector<'_> {
        self.get_vector_opt::<P>(key).expect("vector not found")
    }

    fn read_vectors<P: AccessPattern, U: Copy>(
        &self,
        keys: impl IntoIterator<Item = (U, PointOffsetType)>,
        mut callback: impl FnMut(U, PointOffsetType, CowVector<'_>),
    ) {
        let point_offsets = keys
            .into_iter()
            .map(|(user_data, point_offset)| ((user_data, point_offset), point_offset));

        let vectors =
            super::iter_vectors::<P, _, _, _>(&self.offsets, &self.vectors, point_offsets);

        for ((user_data, point_offset), flattened) in vectors {
            let vector = CowVector::MultiDense(T::into_float_multivector(
                flattened_to_multi_vector(flattened, self.vectors.dim()),
            ));

            callback(user_data, point_offset, vector);
        }
    }

    fn get_vector_opt<P: AccessPattern>(&self, key: PointOffsetType) -> Option<CowVector<'_>> {
        read_multi_vector::<T, P, _>(&self.offsets, &self.vectors, key)
            .map(|multi| CowVector::MultiDense(T::into_float_multivector(multi)))
    }

    fn is_deleted_vector(&self, key: PointOffsetType) -> bool {
        self.deleted.get(key as usize).is_some_and(|bit| *bit)
    }

    fn deleted_vector_count(&self) -> usize {
        self.deleted_count
    }

    fn deleted_vector_bitslice(&self) -> &BitSlice {
        self.deleted.as_bitslice()
    }
}

#[cfg(test)]
mod tests {
    use common::counter::hardware_counter::HardwareCounterCell;
    use common::generic_consts::Random;
    use common::universal_io::{MmapFile, MmapFs};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use tempfile::Builder;

    use super::*;
    use crate::data_types::vectors::{
        MultiDenseVectorInternal, TypedMultiDenseVectorRef, VectorElementType, VectorRef,
    };
    use crate::types::MultiVectorConfig;
    use crate::vector_storage::VectorStorage;
    use crate::vector_storage::multi_dense::appendable_mmap_multi_dense_vector_storage::open_appendable_memmap_multi_vector_storage_impl;

    /// Write multivectors (deleting ~10%) through the writable appendable
    /// storage, then reopen the same directory read-only and assert it mirrors
    /// the state — including per-point multivector contents, which exercises the
    /// offsets storage.
    #[test]
    fn read_only_chunked_multi_dense_round_trip() {
        const POINT_COUNT: PointOffsetType = 1000;
        const DIM: usize = 128;

        let dir = Builder::new().prefix("ro_multi_dense").tempdir().unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let hw = HardwareCounterCell::disposable();

        let multivectors: Vec<MultiDenseVectorInternal> = (0..POINT_COUNT)
            .map(|_| {
                let inner = rng.random_range(1..=4);
                let vectors = std::iter::repeat_with(|| {
                    std::iter::repeat_with(|| rng.random_range(-1.0..1.0))
                        .take(DIM)
                        .collect()
                })
                .take(inner)
                .collect::<Vec<Vec<VectorElementType>>>();
                MultiDenseVectorInternal::try_from(vectors).unwrap()
            })
            .collect();

        let mut deleted_ids = Vec::new();
        {
            let mut storage =
                open_appendable_memmap_multi_vector_storage_impl::<VectorElementType>(
                    dir.path(),
                    DIM,
                    Distance::Dot,
                    MultiVectorConfig::default(),
                    AdviceSetting::Global,
                    false,
                )
                .unwrap();
            for (id, multivec) in multivectors.iter().enumerate() {
                storage
                    .insert_vector(id as PointOffsetType, VectorRef::from(multivec), &hw)
                    .unwrap();
            }
            for id in 0..POINT_COUNT {
                if rng.random_bool(0.1) {
                    storage.delete_vector(id).unwrap();
                    deleted_ids.push(id);
                }
            }
            storage.flusher()().unwrap();
        }

        let storage = ReadOnlyChunkedMultiDenseVectorStorage::<VectorElementType, MmapFile>::open(
            &MmapFs,
            dir.path(),
            DIM,
            Distance::Dot,
            AdviceSetting::Global,
            false,
        )
        .unwrap();

        assert_eq!(storage.total_vector_count(), POINT_COUNT as usize);
        assert_eq!(storage.distance(), Distance::Dot);
        assert_eq!(storage.deleted_vector_count(), deleted_ids.len());

        for id in 0..POINT_COUNT {
            assert_eq!(storage.is_deleted_vector(id), deleted_ids.contains(&id));

            let stored = storage.get_vector::<Random>(id);
            let multi: TypedMultiDenseVectorRef<VectorElementType> =
                stored.as_vec_ref().try_into().unwrap();
            assert_eq!(
                multi.to_owned(),
                multivectors[id as usize],
                "vector {id} mismatch",
            );
        }
    }

    /// A writer appends multivectors and deletes a few; after `live_reload` the
    /// read-only view reflects both.
    #[test]
    fn live_reload_picks_up_appends_and_deletions() {
        const DIM: usize = 48;
        let dir = Builder::new().prefix("ro_multi_reload").tempdir().unwrap();
        let mut rng = StdRng::seed_from_u64(11);
        let hw = HardwareCounterCell::disposable();

        let rand_multi = |rng: &mut StdRng| -> MultiDenseVectorInternal {
            let inner = rng.random_range(1..=3);
            let vectors = std::iter::repeat_with(|| {
                std::iter::repeat_with(|| rng.random_range(-1.0..1.0))
                    .take(DIM)
                    .collect()
            })
            .take(inner)
            .collect::<Vec<Vec<VectorElementType>>>();
            MultiDenseVectorInternal::try_from(vectors).unwrap()
        };
        let first: Vec<MultiDenseVectorInternal> = (0..150).map(|_| rand_multi(&mut rng)).collect();
        let second: Vec<MultiDenseVectorInternal> =
            (0..100).map(|_| rand_multi(&mut rng)).collect();

        let mut writer = open_appendable_memmap_multi_vector_storage_impl::<VectorElementType>(
            dir.path(),
            DIM,
            Distance::Dot,
            MultiVectorConfig::default(),
            AdviceSetting::Global,
            false,
        )
        .unwrap();
        for (id, multivec) in first.iter().enumerate() {
            writer
                .insert_vector(id as PointOffsetType, VectorRef::from(multivec), &hw)
                .unwrap();
        }
        writer.flusher()().unwrap();

        let mut reader =
            ReadOnlyChunkedMultiDenseVectorStorage::<VectorElementType, MmapFile>::open(
                &MmapFs,
                dir.path(),
                DIM,
                Distance::Dot,
                AdviceSetting::Global,
                false,
            )
            .unwrap();
        assert_eq!(reader.total_vector_count(), first.len());

        for (offset, multivec) in second.iter().enumerate() {
            writer
                .insert_vector(
                    (first.len() + offset) as PointOffsetType,
                    VectorRef::from(multivec),
                    &hw,
                )
                .unwrap();
        }
        let deleted_ids: Vec<PointOffsetType> = vec![1, 75, 149];
        for &id in &deleted_ids {
            writer.delete_vector(id).unwrap();
        }
        writer.flusher()().unwrap();

        let new_ids: Vec<PointOffsetType> = (first.len()..first.len() + second.len())
            .map(|offset| offset as PointOffsetType)
            .collect();
        reader
            .live_reload(&MmapFs, &deleted_ids, &new_ids, &hw)
            .unwrap();

        assert_eq!(reader.total_vector_count(), first.len() + second.len());
        assert_eq!(reader.deleted_vector_count(), deleted_ids.len());

        // The appended multivector is visible and correct.
        let stored = reader.get_vector::<Random>(first.len() as PointOffsetType);
        let multi: TypedMultiDenseVectorRef<VectorElementType> =
            stored.as_vec_ref().try_into().unwrap();
        assert_eq!(multi.to_owned(), second[0]);

        for &id in &deleted_ids {
            assert!(reader.is_deleted_vector(id));
        }
        assert!(!reader.is_deleted_vector(0));
    }
}
