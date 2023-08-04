use std::result;
use rust_tokenizers::{error::TokenizerError, tokenizer::{Tokenizer, SentencePieceBpeTokenizer, TruncationStrategy}, vocab::Vocab};

const BOS_TOKEN_ID: i64 = 1;
const EOS_TOKEN_ID: i64 = 2;

pub type Result<T> = result::Result<T, TokenizerError>;

pub struct LlamaTokenizer {
    spm: SentencePieceBpeTokenizer,
}

impl LlamaTokenizer {
    pub fn new(tokenizer_path: &str) -> Result<Self> {
        let lower_case = false;
        SentencePieceBpeTokenizer::from_file(tokenizer_path, lower_case)
            .map(|spm| Self { spm } )
    }

    pub fn encode(&self, text: &str, include_bos: bool, include_eos: bool) -> Vec<i64> {
        let pre = if include_bos {
            vec![BOS_TOKEN_ID]
        } else {
            vec![]
        };

        let post = if include_eos {
            vec![EOS_TOKEN_ID]
        } else {
            vec![]
        };

        let token_ids = self.spm.encode(text, None, std::usize::MAX, &TruncationStrategy::LongestFirst, 0).token_ids;

        [pre, token_ids, post]
            .into_iter()
            .flat_map(|v| v.into_iter())
            .collect()
    }

    pub fn decode(&self, tokens: &[i64], skip_special_tokens: bool) -> String {
        let clean_spaces = false;
        self.spm.decode(tokens, skip_special_tokens, clean_spaces)
    }

    pub fn vocab_size(&self, include_special_tokens: bool) -> usize {
        let vocab = self.spm.vocab();
        if include_special_tokens {
            vocab.values().len() + vocab.special_values().len()
        } else {
            vocab.values().len()
        }
    }
}
