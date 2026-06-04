use std::borrow::Cow;
use std::cmp::min;
use std::iter::Iterator;
use std::ops::Range;
use std::sync::atomic::AtomicBool;

use ahash::AHashMap;
use common::generic_consts::Sequential;
use common::small_uint::U24;
use common::types::PointOffsetType;
use sparse::common::sparse_vector::SparseVector;

use crate::common::operation_error::{OperationError, OperationResult};
use crate::data_types::named_vectors::{CowMultiVector, CowVector};
use crate::data_types::vectors::{VectorElementType, VectorElementTypeByte, VectorElementTypeHalf};
use crate::types::CompactExtendedPointId;
use crate::vector_storage::{
    DenseVectorStorage, MultiVectorStorage, SparseVectorStorage, VectorStorageEnum,
    VectorStorageRead,
};

const BATCH_SIZE: usize = 256;

/// Define location of the point source during segment construction.
pub struct PointData {
    pub external_id: CompactExtendedPointId,
    /// [`CompactExtendedPointId`] is 17 bytes, we reduce
    /// `segment_index` to 3 bytes to avoid paddings and align nicely.
    pub segment_index: U24,
    pub internal_id: PointOffsetType,
    pub version: u64,
    pub ordering: u64,
}

/// Reads a point's vector (in the native representation `V`) and its deleted
/// flag from a source storage. One reader per kind/element-type — all source
/// variants of that kind are accepted, so e.g. a volatile source can feed an
/// mmap target without an f32 round-trip.
type ReadFn<'a, V> = fn(&'a VectorStorageEnum, PointOffsetType) -> (V, bool);

/// Batched iterator over points to insert, reading each source vector in its
/// **native** representation `V` — no `f32` round-trip.
///
/// Reads `BATCH_SIZE` points into a buffer (grouped by source segment for read
/// locality) and then iterates over them.
pub struct BatchedReader<'a, V> {
    points: &'a [PointData],
    sources: &'a [&'a VectorStorageEnum],
    read: ReadFn<'a, V>,
    buffer: Vec<Option<(V, bool)>>,
    seg_to_points_buffer: AHashMap<U24, Vec<(&'a PointData, usize)>>,
    /// Global position of the iterator.
    /// From 0 to `points.len()`.
    position: usize,
}

impl<'a, V> BatchedReader<'a, V> {
    pub fn new(
        points: &'a [PointData],
        sources: &'a [&'a VectorStorageEnum],
        read: ReadFn<'a, V>,
    ) -> BatchedReader<'a, V> {
        // We don't know the vector type's size, so pre-size the buffer with
        // empty slots and fill them per batch.
        let buffer = (0..BATCH_SIZE).map(|_| None).collect();

        BatchedReader {
            points,
            sources,
            read,
            buffer,
            seg_to_points_buffer: AHashMap::default(),
            position: 0,
        }
    }

    /// Fills the buffer with the next batch of points.
    fn refill_buffer(&mut self) {
        let start_pos = self.position;
        let end_pos = min(self.position + BATCH_SIZE, self.points.len());

        // Read by segments, as we want to localize reads as much as possible.
        for pos in start_pos..end_pos {
            let point_data = &self.points[pos];
            let offset_in_batch = pos - start_pos;

            self.seg_to_points_buffer
                .entry(point_data.segment_index)
                .or_default()
                .push((point_data, offset_in_batch))
        }

        for (segment_index, points) in self.seg_to_points_buffer.drain() {
            let source = self.sources[segment_index.get() as usize];
            for (point_data, offset_in_batch) in points {
                self.buffer[offset_in_batch] = Some((self.read)(source, point_data.internal_id));
            }
        }
    }

    fn refill_buffer_if_needed(&mut self) {
        if self.position.is_multiple_of(BATCH_SIZE) {
            self.refill_buffer();
        }
    }
}

impl<'a, V> Iterator for BatchedReader<'a, V> {
    type Item = (V, bool);

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.points.len() {
            return None;
        }

        self.refill_buffer_if_needed();

        let item = self.buffer[self.position % BATCH_SIZE]
            .take()
            .expect("buffer slot must be filled for a valid position");
        self.position += 1;

        Some(item)
    }
}

/// Vector kind and element type a storage merges as. Storages within one group
/// are interchangeable on read (e.g. volatile vs mmap vs appendable), so a
/// merge is only valid between storages of the same group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MergeGroup {
    DenseF32,
    DenseByte,
    DenseHalf,
    MultiF32,
    MultiByte,
    MultiHalf,
    Sparse,
}

fn merge_group(storage: &VectorStorageEnum) -> MergeGroup {
    match storage {
        VectorStorageEnum::DenseVolatile(_) => MergeGroup::DenseF32,
        #[cfg(test)]
        VectorStorageEnum::DenseVolatileByte(_) => MergeGroup::DenseByte,
        #[cfg(test)]
        VectorStorageEnum::DenseVolatileHalf(_) => MergeGroup::DenseHalf,
        VectorStorageEnum::DenseMemmap(_) => MergeGroup::DenseF32,
        VectorStorageEnum::DenseMemmapByte(_) => MergeGroup::DenseByte,
        VectorStorageEnum::DenseMemmapHalf(_) => MergeGroup::DenseHalf,

        #[cfg(target_os = "linux")]
        VectorStorageEnum::DenseUring(_) => MergeGroup::DenseF32,
        #[cfg(target_os = "linux")]
        VectorStorageEnum::DenseUringByte(_) => MergeGroup::DenseByte,
        #[cfg(target_os = "linux")]
        VectorStorageEnum::DenseUringHalf(_) => MergeGroup::DenseHalf,

        VectorStorageEnum::DenseAppendableMemmap(_) => MergeGroup::DenseF32,
        VectorStorageEnum::DenseAppendableMemmapByte(_) => MergeGroup::DenseByte,
        VectorStorageEnum::DenseAppendableMemmapHalf(_) => MergeGroup::DenseHalf,
        VectorStorageEnum::SparseVolatile(_) => MergeGroup::Sparse,
        VectorStorageEnum::SparseMmap(_) => MergeGroup::Sparse,
        VectorStorageEnum::MultiDenseVolatile(_) => MergeGroup::MultiF32,
        #[cfg(test)]
        VectorStorageEnum::MultiDenseVolatileByte(_) => MergeGroup::MultiByte,
        #[cfg(test)]
        VectorStorageEnum::MultiDenseVolatileHalf(_) => MergeGroup::MultiHalf,
        VectorStorageEnum::MultiDenseAppendableMemmap(_) => MergeGroup::MultiF32,
        VectorStorageEnum::MultiDenseAppendableMemmapByte(_) => MergeGroup::MultiByte,
        VectorStorageEnum::MultiDenseAppendableMemmapHalf(_) => MergeGroup::MultiHalf,
        VectorStorageEnum::EmptyDense(_) => MergeGroup::DenseF32,
        VectorStorageEnum::EmptySparse(_) => MergeGroup::Sparse,
    }
}

// Native source readers, one per (kind, element type). All variants of the
// same kind/element type are accepted, since storage variants of one element
// type are interchangeable on read (e.g. volatile vs mmap vs appendable).
//
// `merge_from` validates that every source matches the target's group up front,
// so the catch-all arms below are unreachable.

fn read_dense_f32(
    source: &VectorStorageEnum,
    key: PointOffsetType,
) -> (Cow<'_, [VectorElementType]>, bool) {
    let deleted = source.is_deleted_vector(key);
    let vector = match source {
        VectorStorageEnum::DenseVolatile(v) => v.get_dense::<Sequential>(key),
        VectorStorageEnum::DenseMemmap(v) => v.get_dense::<Sequential>(key),
        #[cfg(target_os = "linux")]
        VectorStorageEnum::DenseUring(v) => v.get_dense::<Sequential>(key),
        VectorStorageEnum::DenseAppendableMemmap(v) => v.get_dense::<Sequential>(key),
        VectorStorageEnum::EmptyDense(v) => v.get_dense::<Sequential>(key),
        VectorStorageEnum::DenseMemmapByte(_)
        | VectorStorageEnum::DenseMemmapHalf(_)
        | VectorStorageEnum::DenseAppendableMemmapByte(_)
        | VectorStorageEnum::DenseAppendableMemmapHalf(_)
        | VectorStorageEnum::SparseVolatile(_)
        | VectorStorageEnum::SparseMmap(_)
        | VectorStorageEnum::MultiDenseVolatile(_)
        | VectorStorageEnum::MultiDenseAppendableMemmap(_)
        | VectorStorageEnum::MultiDenseAppendableMemmapByte(_)
        | VectorStorageEnum::MultiDenseAppendableMemmapHalf(_)
        | VectorStorageEnum::EmptySparse(_) => {
            unreachable!("source group validated by merge_from")
        }
        #[cfg(test)]
        VectorStorageEnum::DenseVolatileByte(_)
        | VectorStorageEnum::DenseVolatileHalf(_)
        | VectorStorageEnum::MultiDenseVolatileByte(_)
        | VectorStorageEnum::MultiDenseVolatileHalf(_) => {
            unreachable!("source group validated by merge_from")
        }
        #[cfg(target_os = "linux")]
        VectorStorageEnum::DenseUringByte(_) | VectorStorageEnum::DenseUringHalf(_) => {
            unreachable!("source group validated by merge_from")
        }
    };
    (vector, deleted)
}

fn read_dense_byte(
    source: &VectorStorageEnum,
    key: PointOffsetType,
) -> (Cow<'_, [VectorElementTypeByte]>, bool) {
    let deleted = source.is_deleted_vector(key);
    let vector = match source {
        #[cfg(test)]
        VectorStorageEnum::DenseVolatileByte(v) => v.get_dense::<Sequential>(key),
        VectorStorageEnum::DenseMemmapByte(v) => v.get_dense::<Sequential>(key),
        #[cfg(target_os = "linux")]
        VectorStorageEnum::DenseUringByte(v) => v.get_dense::<Sequential>(key),
        VectorStorageEnum::DenseAppendableMemmapByte(v) => v.get_dense::<Sequential>(key),
        VectorStorageEnum::DenseVolatile(_)
        | VectorStorageEnum::DenseMemmap(_)
        | VectorStorageEnum::DenseMemmapHalf(_)
        | VectorStorageEnum::DenseAppendableMemmap(_)
        | VectorStorageEnum::DenseAppendableMemmapHalf(_)
        | VectorStorageEnum::SparseVolatile(_)
        | VectorStorageEnum::SparseMmap(_)
        | VectorStorageEnum::MultiDenseVolatile(_)
        | VectorStorageEnum::MultiDenseAppendableMemmap(_)
        | VectorStorageEnum::MultiDenseAppendableMemmapByte(_)
        | VectorStorageEnum::MultiDenseAppendableMemmapHalf(_)
        | VectorStorageEnum::EmptyDense(_)
        | VectorStorageEnum::EmptySparse(_) => {
            unreachable!("source group validated by merge_from")
        }
        #[cfg(test)]
        VectorStorageEnum::DenseVolatileHalf(_)
        | VectorStorageEnum::MultiDenseVolatileByte(_)
        | VectorStorageEnum::MultiDenseVolatileHalf(_) => {
            unreachable!("source group validated by merge_from")
        }
        #[cfg(target_os = "linux")]
        VectorStorageEnum::DenseUring(_) | VectorStorageEnum::DenseUringHalf(_) => {
            unreachable!("source group validated by merge_from")
        }
    };
    (vector, deleted)
}

fn read_dense_half(
    source: &VectorStorageEnum,
    key: PointOffsetType,
) -> (Cow<'_, [VectorElementTypeHalf]>, bool) {
    let deleted = source.is_deleted_vector(key);
    let vector = match source {
        #[cfg(test)]
        VectorStorageEnum::DenseVolatileHalf(v) => v.get_dense::<Sequential>(key),
        VectorStorageEnum::DenseMemmapHalf(v) => v.get_dense::<Sequential>(key),
        #[cfg(target_os = "linux")]
        VectorStorageEnum::DenseUringHalf(v) => v.get_dense::<Sequential>(key),
        VectorStorageEnum::DenseAppendableMemmapHalf(v) => v.get_dense::<Sequential>(key),
        VectorStorageEnum::DenseVolatile(_)
        | VectorStorageEnum::DenseMemmap(_)
        | VectorStorageEnum::DenseMemmapByte(_)
        | VectorStorageEnum::DenseAppendableMemmap(_)
        | VectorStorageEnum::DenseAppendableMemmapByte(_)
        | VectorStorageEnum::SparseVolatile(_)
        | VectorStorageEnum::SparseMmap(_)
        | VectorStorageEnum::MultiDenseVolatile(_)
        | VectorStorageEnum::MultiDenseAppendableMemmap(_)
        | VectorStorageEnum::MultiDenseAppendableMemmapByte(_)
        | VectorStorageEnum::MultiDenseAppendableMemmapHalf(_)
        | VectorStorageEnum::EmptyDense(_)
        | VectorStorageEnum::EmptySparse(_) => {
            unreachable!("source group validated by merge_from")
        }
        #[cfg(test)]
        VectorStorageEnum::DenseVolatileByte(_)
        | VectorStorageEnum::MultiDenseVolatileByte(_)
        | VectorStorageEnum::MultiDenseVolatileHalf(_) => {
            unreachable!("source group validated by merge_from")
        }
        #[cfg(target_os = "linux")]
        VectorStorageEnum::DenseUring(_) | VectorStorageEnum::DenseUringByte(_) => {
            unreachable!("source group validated by merge_from")
        }
    };
    (vector, deleted)
}

fn read_multi_f32(
    source: &VectorStorageEnum,
    key: PointOffsetType,
) -> (CowMultiVector<'_, VectorElementType>, bool) {
    let deleted = source.is_deleted_vector(key);
    let vector = match source {
        VectorStorageEnum::MultiDenseVolatile(v) => v.get_multi::<Sequential>(key),
        VectorStorageEnum::MultiDenseAppendableMemmap(v) => v.get_multi::<Sequential>(key),
        VectorStorageEnum::DenseVolatile(_)
        | VectorStorageEnum::DenseMemmap(_)
        | VectorStorageEnum::DenseMemmapByte(_)
        | VectorStorageEnum::DenseMemmapHalf(_)
        | VectorStorageEnum::DenseAppendableMemmap(_)
        | VectorStorageEnum::DenseAppendableMemmapByte(_)
        | VectorStorageEnum::DenseAppendableMemmapHalf(_)
        | VectorStorageEnum::SparseVolatile(_)
        | VectorStorageEnum::SparseMmap(_)
        | VectorStorageEnum::MultiDenseAppendableMemmapByte(_)
        | VectorStorageEnum::MultiDenseAppendableMemmapHalf(_)
        | VectorStorageEnum::EmptyDense(_)
        | VectorStorageEnum::EmptySparse(_) => {
            unreachable!("source group validated by merge_from")
        }
        #[cfg(test)]
        VectorStorageEnum::DenseVolatileByte(_)
        | VectorStorageEnum::DenseVolatileHalf(_)
        | VectorStorageEnum::MultiDenseVolatileByte(_)
        | VectorStorageEnum::MultiDenseVolatileHalf(_) => {
            unreachable!("source group validated by merge_from")
        }
        #[cfg(target_os = "linux")]
        VectorStorageEnum::DenseUring(_)
        | VectorStorageEnum::DenseUringByte(_)
        | VectorStorageEnum::DenseUringHalf(_) => {
            unreachable!("source group validated by merge_from")
        }
    };
    (vector, deleted)
}

fn read_multi_byte(
    source: &VectorStorageEnum,
    key: PointOffsetType,
) -> (CowMultiVector<'_, VectorElementTypeByte>, bool) {
    let deleted = source.is_deleted_vector(key);
    let vector = match source {
        #[cfg(test)]
        VectorStorageEnum::MultiDenseVolatileByte(v) => v.get_multi::<Sequential>(key),
        VectorStorageEnum::MultiDenseAppendableMemmapByte(v) => v.get_multi::<Sequential>(key),
        VectorStorageEnum::DenseVolatile(_)
        | VectorStorageEnum::DenseMemmap(_)
        | VectorStorageEnum::DenseMemmapByte(_)
        | VectorStorageEnum::DenseMemmapHalf(_)
        | VectorStorageEnum::DenseAppendableMemmap(_)
        | VectorStorageEnum::DenseAppendableMemmapByte(_)
        | VectorStorageEnum::DenseAppendableMemmapHalf(_)
        | VectorStorageEnum::SparseVolatile(_)
        | VectorStorageEnum::SparseMmap(_)
        | VectorStorageEnum::MultiDenseVolatile(_)
        | VectorStorageEnum::MultiDenseAppendableMemmap(_)
        | VectorStorageEnum::MultiDenseAppendableMemmapHalf(_)
        | VectorStorageEnum::EmptyDense(_)
        | VectorStorageEnum::EmptySparse(_) => {
            unreachable!("source group validated by merge_from")
        }
        #[cfg(test)]
        VectorStorageEnum::DenseVolatileByte(_)
        | VectorStorageEnum::DenseVolatileHalf(_)
        | VectorStorageEnum::MultiDenseVolatileHalf(_) => {
            unreachable!("source group validated by merge_from")
        }
        #[cfg(target_os = "linux")]
        VectorStorageEnum::DenseUring(_)
        | VectorStorageEnum::DenseUringByte(_)
        | VectorStorageEnum::DenseUringHalf(_) => {
            unreachable!("source group validated by merge_from")
        }
    };
    (vector, deleted)
}

fn read_multi_half(
    source: &VectorStorageEnum,
    key: PointOffsetType,
) -> (CowMultiVector<'_, VectorElementTypeHalf>, bool) {
    let deleted = source.is_deleted_vector(key);
    let vector = match source {
        #[cfg(test)]
        VectorStorageEnum::MultiDenseVolatileHalf(v) => v.get_multi::<Sequential>(key),
        VectorStorageEnum::MultiDenseAppendableMemmapHalf(v) => v.get_multi::<Sequential>(key),
        VectorStorageEnum::DenseVolatile(_)
        | VectorStorageEnum::DenseMemmap(_)
        | VectorStorageEnum::DenseMemmapByte(_)
        | VectorStorageEnum::DenseMemmapHalf(_)
        | VectorStorageEnum::DenseAppendableMemmap(_)
        | VectorStorageEnum::DenseAppendableMemmapByte(_)
        | VectorStorageEnum::DenseAppendableMemmapHalf(_)
        | VectorStorageEnum::SparseVolatile(_)
        | VectorStorageEnum::SparseMmap(_)
        | VectorStorageEnum::MultiDenseVolatile(_)
        | VectorStorageEnum::MultiDenseAppendableMemmap(_)
        | VectorStorageEnum::MultiDenseAppendableMemmapByte(_)
        | VectorStorageEnum::EmptyDense(_)
        | VectorStorageEnum::EmptySparse(_) => {
            unreachable!("source group validated by merge_from")
        }
        #[cfg(test)]
        VectorStorageEnum::DenseVolatileByte(_)
        | VectorStorageEnum::DenseVolatileHalf(_)
        | VectorStorageEnum::MultiDenseVolatileByte(_) => {
            unreachable!("source group validated by merge_from")
        }
        #[cfg(target_os = "linux")]
        VectorStorageEnum::DenseUring(_)
        | VectorStorageEnum::DenseUringByte(_)
        | VectorStorageEnum::DenseUringHalf(_) => {
            unreachable!("source group validated by merge_from")
        }
    };
    (vector, deleted)
}

fn read_sparse(source: &VectorStorageEnum, key: PointOffsetType) -> (Cow<'_, SparseVector>, bool) {
    let deleted = source.is_deleted_vector(key);
    // Sparse has no f32 round-trip to avoid, so reuse the generic read path.
    let vector = match source.get_vector::<Sequential>(key) {
        CowVector::Sparse(v) => v,
        CowVector::Dense(_) | CowVector::MultiDense(_) => {
            unreachable!("sparse vector storage returned a non-sparse vector")
        }
    };
    (vector, deleted)
}

/// Append `points` (read from `sources`) into `target`, copying each vector in
/// its native representation — no `f32` round-trip, no dequantization.
///
/// Every source must be the same vector kind and element type as `target`; the
/// storage type is fixed by the collection config, so this always holds within
/// one merge. A mismatch returns a service error.
pub fn merge_from<'a>(
    target: &mut VectorStorageEnum,
    points: &'a [PointData],
    sources: &'a [&'a VectorStorageEnum],
    stopped: &AtomicBool,
) -> OperationResult<Range<PointOffsetType>> {
    // Validate up front that every source is the same kind/element type as the
    // target, so the readers below can copy data natively without conversion.
    let target_group = merge_group(target);
    for (index, source) in sources.iter().enumerate() {
        let source_group = merge_group(source);
        if source_group != target_group {
            return Err(OperationError::service_error(format!(
                "Cannot merge vector storage: source #{index} is {source_group:?}, \
                 but the target is {target_group:?}"
            )));
        }
    }

    match target {
        VectorStorageEnum::DenseVolatile(target) => target.update_from(
            &mut BatchedReader::new(points, sources, read_dense_f32),
            stopped,
        ),
        #[cfg(test)]
        VectorStorageEnum::DenseVolatileByte(target) => target.update_from(
            &mut BatchedReader::new(points, sources, read_dense_byte),
            stopped,
        ),
        #[cfg(test)]
        VectorStorageEnum::DenseVolatileHalf(target) => target.update_from(
            &mut BatchedReader::new(points, sources, read_dense_half),
            stopped,
        ),
        VectorStorageEnum::DenseMemmap(target) => target.update_from(
            &mut BatchedReader::new(points, sources, read_dense_f32),
            stopped,
        ),
        VectorStorageEnum::DenseMemmapByte(target) => target.update_from(
            &mut BatchedReader::new(points, sources, read_dense_byte),
            stopped,
        ),
        VectorStorageEnum::DenseMemmapHalf(target) => target.update_from(
            &mut BatchedReader::new(points, sources, read_dense_half),
            stopped,
        ),

        #[cfg(target_os = "linux")]
        VectorStorageEnum::DenseUring(target) => target.update_from(
            &mut BatchedReader::new(points, sources, read_dense_f32),
            stopped,
        ),
        #[cfg(target_os = "linux")]
        VectorStorageEnum::DenseUringByte(target) => target.update_from(
            &mut BatchedReader::new(points, sources, read_dense_byte),
            stopped,
        ),
        #[cfg(target_os = "linux")]
        VectorStorageEnum::DenseUringHalf(target) => target.update_from(
            &mut BatchedReader::new(points, sources, read_dense_half),
            stopped,
        ),

        VectorStorageEnum::DenseAppendableMemmap(target) => target.update_from(
            &mut BatchedReader::new(points, sources, read_dense_f32),
            stopped,
        ),
        VectorStorageEnum::DenseAppendableMemmapByte(target) => target.update_from(
            &mut BatchedReader::new(points, sources, read_dense_byte),
            stopped,
        ),
        VectorStorageEnum::DenseAppendableMemmapHalf(target) => target.update_from(
            &mut BatchedReader::new(points, sources, read_dense_half),
            stopped,
        ),
        VectorStorageEnum::SparseVolatile(target) => target.update_from(
            &mut BatchedReader::new(points, sources, read_sparse),
            stopped,
        ),
        VectorStorageEnum::SparseMmap(target) => target.update_from(
            &mut BatchedReader::new(points, sources, read_sparse),
            stopped,
        ),
        VectorStorageEnum::MultiDenseVolatile(target) => target.update_from(
            &mut BatchedReader::new(points, sources, read_multi_f32),
            stopped,
        ),
        #[cfg(test)]
        VectorStorageEnum::MultiDenseVolatileByte(target) => target.update_from(
            &mut BatchedReader::new(points, sources, read_multi_byte),
            stopped,
        ),
        #[cfg(test)]
        VectorStorageEnum::MultiDenseVolatileHalf(target) => target.update_from(
            &mut BatchedReader::new(points, sources, read_multi_half),
            stopped,
        ),
        VectorStorageEnum::MultiDenseAppendableMemmap(target) => target.update_from(
            &mut BatchedReader::new(points, sources, read_multi_f32),
            stopped,
        ),
        VectorStorageEnum::MultiDenseAppendableMemmapByte(target) => target.update_from(
            &mut BatchedReader::new(points, sources, read_multi_byte),
            stopped,
        ),
        VectorStorageEnum::MultiDenseAppendableMemmapHalf(target) => target.update_from(
            &mut BatchedReader::new(points, sources, read_multi_half),
            stopped,
        ),
        VectorStorageEnum::EmptyDense(target) => target.update_from(
            &mut BatchedReader::new(points, sources, read_dense_f32),
            stopped,
        ),
        VectorStorageEnum::EmptySparse(target) => target.update_from(
            &mut BatchedReader::new(points, sources, read_sparse),
            stopped,
        ),
    }
}

/// Test-only helper: merge `count` points (offsets `0..count`) from a single
/// `source` storage into `target` via [`merge_from`].
#[cfg(test)]
pub fn merge_from_single_source(
    target: &mut VectorStorageEnum,
    source: &VectorStorageEnum,
    count: PointOffsetType,
) -> OperationResult<Range<PointOffsetType>> {
    use crate::types::PointIdType;

    let points: Vec<PointData> = (0..count)
        .map(|internal_id| PointData {
            external_id: CompactExtendedPointId::from(PointIdType::NumId(internal_id as u64)),
            segment_index: U24::new_wrapped(0),
            internal_id,
            version: 0,
            ordering: 0,
        })
        .collect();
    let sources = [source];
    merge_from(target, &points, &sources, &AtomicBool::default())
}
