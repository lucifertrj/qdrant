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
use crate::vector_storage::chunked_vectors::ChunkedVectorsRead;
use crate::vector_storage::dense::appendable_dense_vector_storage::{
    DELETED_DIR_PATH, VECTORS_DIR_PATH,
};
use crate::vector_storage::{VectorOffsetType, VectorStorageRead};

#[derive(Debug)]
pub struct ReadOnlyChunkedDenseVectorStorage<T: PrimitiveVectorElement, S: UniversalRead> {
    vectors: ChunkedVectorsRead<T, S>,
    /// Flags marking deleted vectors
    ///
    /// Structure grows dynamically, but may be smaller than actual number of vectors. Must not
    /// depend on its length.
    deleted: BitVec,
    distance: Distance,
    deleted_count: usize,
    /// Chunk-open settings retained so [`LiveReload`] can refresh the chunked
    /// vectors with the same residency.
    advice: AdviceSetting,
    populate: bool,
}

impl<T: PrimitiveVectorElement, S: UniversalRead> ReadOnlyChunkedDenseVectorStorage<T, S> {
    /// Open the read-only counterpart of [`AppendableMmapDenseVectorStorage`][1]
    /// at `path`, threading every file open through `fs`.
    ///
    /// Reads the same on-disk layout the writable storage maintains — the
    /// chunked `vectors/` directory and the `deleted/` flags — but never creates
    /// or writes anything. The deleted flags are materialized into an owned
    /// bitvec via [`DynamicStoredFlags::load_bitvec`]. `populate` warms the
    /// vector chunks (mirroring `is_on_disk = !populate`); it does not apply to
    /// the always-resident deleted bitvec.
    ///
    /// [1]: super::super::appendable_dense_vector_storage::AppendableMmapDenseVectorStorage
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

        let deleted = DynamicStoredFlags::<S>::load_bitvec(fs, &path.join(DELETED_DIR_PATH))?;
        let deleted_count = deleted.count_ones();

        Ok(Self {
            vectors,
            deleted,
            distance,
            deleted_count,
            advice,
            populate,
        })
    }
}

impl<T: PrimitiveVectorElement, S: UniversalRead> LiveReload
    for ReadOnlyChunkedDenseVectorStorage<T, S>
{
    type Fs = S::Fs;

    /// Refresh from disk: reload the chunked vectors (picking up appended
    /// vectors and new chunks) and apply the authoritative `deleted_points` to
    /// the in-memory deletion bitvec. Newly added points are served straight
    /// from the refreshed chunks, so `new_points` is unused.
    fn live_reload(
        &mut self,
        fs: &S::Fs,
        deleted_points: &[PointOffsetType],
        _new_points: &[PointOffsetType],
        _hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        self.vectors.live_reload(fs, self.advice, self.populate)?;

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
    for ReadOnlyChunkedDenseVectorStorage<T, S>
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
        self.vectors.len()
    }

    fn get_vector<P: AccessPattern>(&self, key: PointOffsetType) -> CowVector<'_> {
        self.vectors
            .get::<P>(key as VectorOffsetType)
            .map(|slice| CowVector::from(T::slice_to_float_cow(slice)))
            .expect("Vector not found")
    }

    fn read_vectors<P: AccessPattern, U: Copy>(
        &self,
        keys: impl IntoIterator<Item = (U, PointOffsetType)>,
        mut callback: impl FnMut(U, PointOffsetType, CowVector<'_>),
    ) {
        let keys = keys
            .into_iter()
            .map(|(user_data, point_offset)| ((user_data, point_offset), point_offset, 1));

        for ((user_data, point_offset), vector) in self.vectors.iter_vectors::<P, _>(keys) {
            let vector = CowVector::from(T::slice_to_float_cow(vector));
            callback(user_data, point_offset, vector);
        }
    }

    fn get_vector_opt<P: AccessPattern>(&self, key: PointOffsetType) -> Option<CowVector<'_>> {
        self.vectors
            .get::<P>(key as VectorOffsetType)
            .map(|slice| CowVector::from(T::slice_to_float_cow(slice)))
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
    use crate::data_types::vectors::{DenseVector, VectorElementType, VectorRef};
    use crate::vector_storage::VectorStorage;
    use crate::vector_storage::dense::appendable_dense_vector_storage::open_appendable_memmap_vector_storage_impl;

    /// Write vectors (deleting ~10%) through the writable appendable storage,
    /// then reopen the same directory read-only and assert it mirrors the state.
    #[test]
    fn read_only_chunked_dense_round_trip() {
        const POINT_COUNT: PointOffsetType = 2500; // spans more than one chunk
        const DIM: usize = 128;

        let dir = Builder::new().prefix("ro_dense").tempdir().unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let hw = HardwareCounterCell::disposable();

        let vectors: Vec<DenseVector> = (0..POINT_COUNT)
            .map(|_| {
                std::iter::repeat_with(|| rng.random_range(-1.0..1.0))
                    .take(DIM)
                    .collect()
            })
            .collect();

        let mut deleted_ids = Vec::new();
        {
            let mut storage = open_appendable_memmap_vector_storage_impl::<VectorElementType>(
                dir.path(),
                DIM,
                Distance::Dot,
                AdviceSetting::Global,
                false,
            )
            .unwrap();
            for (id, vector) in vectors.iter().enumerate() {
                storage
                    .insert_vector(id as PointOffsetType, VectorRef::from(vector), &hw)
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

        let storage = ReadOnlyChunkedDenseVectorStorage::<VectorElementType, MmapFile>::open(
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

            let got: DenseVector = storage
                .get_vector::<Random>(id)
                .to_owned()
                .try_into()
                .unwrap();
            assert_eq!(got, vectors[id as usize], "vector {id} mismatch");
        }
    }

    /// A writer appends vectors and deletes a few; after `live_reload` the
    /// read-only view reflects both.
    #[test]
    fn live_reload_picks_up_appends_and_deletions() {
        const DIM: usize = 64;
        let dir = Builder::new().prefix("ro_dense_reload").tempdir().unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        let hw = HardwareCounterCell::disposable();

        let rand_vec = |rng: &mut StdRng| -> DenseVector {
            std::iter::repeat_with(|| rng.random_range(-1.0..1.0))
                .take(DIM)
                .collect()
        };
        let first: Vec<DenseVector> = (0..200).map(|_| rand_vec(&mut rng)).collect();
        let second: Vec<DenseVector> = (0..150).map(|_| rand_vec(&mut rng)).collect();

        let mut writer = open_appendable_memmap_vector_storage_impl::<VectorElementType>(
            dir.path(),
            DIM,
            Distance::Dot,
            AdviceSetting::Global,
            false,
        )
        .unwrap();
        for (id, vector) in first.iter().enumerate() {
            writer
                .insert_vector(id as PointOffsetType, VectorRef::from(vector), &hw)
                .unwrap();
        }
        writer.flusher()().unwrap();

        let mut reader = ReadOnlyChunkedDenseVectorStorage::<VectorElementType, MmapFile>::open(
            &MmapFs,
            dir.path(),
            DIM,
            Distance::Dot,
            AdviceSetting::Global,
            false,
        )
        .unwrap();
        assert_eq!(reader.total_vector_count(), first.len());

        // Append new vectors and delete a few existing ones through the writer.
        for (offset, vector) in second.iter().enumerate() {
            writer
                .insert_vector(
                    (first.len() + offset) as PointOffsetType,
                    VectorRef::from(vector),
                    &hw,
                )
                .unwrap();
        }
        let deleted_ids: Vec<PointOffsetType> = vec![3, 50, 199];
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

        // The appended vector is now visible and correct.
        let got: DenseVector = reader
            .get_vector::<Random>(first.len() as PointOffsetType)
            .to_owned()
            .try_into()
            .unwrap();
        assert_eq!(got, second[0]);

        // Deletions are reflected; untouched points stay live.
        for &id in &deleted_ids {
            assert!(reader.is_deleted_vector(id));
        }
        assert!(!reader.is_deleted_vector(0));
    }
}
