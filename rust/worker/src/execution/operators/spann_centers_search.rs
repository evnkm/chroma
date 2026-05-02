use async_trait::async_trait;
use chroma_error::{ChromaError, ErrorCodes};
use chroma_index::spann::head_bloom::EqualityTokens;
use chroma_segment::distributed_spann::SpannSegmentReaderShard;
use chroma_system::Operator;
use thiserror::Error;

#[derive(Debug)]
pub(crate) struct SpannCentersSearchInput<'referred_data> {
    pub(crate) reader: Option<SpannSegmentReaderShard<'referred_data>>,
    // Assumes that query is already normalized in case of cosine.
    pub(crate) normalized_query: Vec<f32>,
    pub(crate) collection_num_records_post_compaction: usize,
    pub(crate) k: usize,
    // Fraction of compacted records that pass the metadata filter, in [0, 1].
    // None means no filter / unknown — adaptive boost is skipped.
    pub(crate) filter_selectivity: Option<f64>,
    pub(crate) head_bloom_tokens: EqualityTokens,
}

#[derive(Debug)]
pub(crate) struct SpannCentersSearchOutput {
    pub(crate) center_ids: Vec<usize>,
    pub(crate) heads_rng: usize,
    pub(crate) heads_after_bloom: usize,
}

#[derive(Error, Debug)]
pub enum SpannCentersSearchError {
    #[error("Error creating spann segment reader")]
    SpannSegmentReaderShardCreationError,
    #[error("Error querying RNG")]
    RngQueryError,
}

impl ChromaError for SpannCentersSearchError {
    fn code(&self) -> ErrorCodes {
        match self {
            Self::SpannSegmentReaderShardCreationError => ErrorCodes::Internal,
            Self::RngQueryError => ErrorCodes::Internal,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SpannCentersSearchOperator {}

#[async_trait]
impl Operator<SpannCentersSearchInput<'_>, SpannCentersSearchOutput>
    for SpannCentersSearchOperator
{
    type Error = SpannCentersSearchError;

    async fn run(
        &self,
        input: &SpannCentersSearchInput,
    ) -> Result<SpannCentersSearchOutput, SpannCentersSearchError> {
        match &input.reader {
            Some(reader) => {
                // Use the reader to query the centers.
                let res = reader
                    .rng_query(
                        &input.normalized_query,
                        input.collection_num_records_post_compaction,
                        input.k,
                        input.filter_selectivity,
                    )
                    .await
                    .map_err(|_| SpannCentersSearchError::RngQueryError)?;
                let center_ids = res.0;
                let heads_rng = center_ids.len();
                let center_ids = reader.gate_heads(&center_ids, &input.head_bloom_tokens);
                let heads_after_bloom = center_ids.len();
                Ok(SpannCentersSearchOutput {
                    center_ids,
                    heads_rng,
                    heads_after_bloom,
                })
            }
            None => Err(SpannCentersSearchError::SpannSegmentReaderShardCreationError),
        }
    }
}
