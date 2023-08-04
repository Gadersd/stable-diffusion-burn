use std::collections::HashMap;
use regex::Regex;

use std::fs::File;
use std::io::{self, BufRead};

fn bytes_to_unicode() -> Vec<(u8, char)> {
    let mut bs: Vec<u8> = ('!' as u8 ..= '~' as u8).into_iter()
        .chain( ('¡' as u8..='¬' as u8).into_iter() )
        .chain( ('®' as u8..='ÿ' as u8).into_iter() )
        .collect();

    let mut cs: Vec<_> = bs.iter().cloned().map(char::from).collect();

    let mut n = 0;
    for b in 0u8..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push( char::from_u32(256 + n).unwrap() );
            n += 1;
        }
    }

    bs.into_iter()
        .zip(
            cs.into_iter()
                .map(|c| c.into())
        ).collect()
}

fn get_pairs(word: &[String]) -> Vec<(String, String)> {
    let prev = word.into_iter().cloned();
    let next = prev.clone().skip(1);

    prev
        .zip(next)
        .collect()
}

fn whitespace_clean(text: &str) -> String {
    text.split_whitespace().collect::<Vec<&str>>().join(" ")
}

fn load_merges(path: &str) -> io::Result<Vec<(String, String)>> {
    let file = File::open(&path)?;
    let reader = io::BufReader::new(file);
    
    let mut merges = Vec::new();
    
    for line in reader.lines() {
        let line = line?;
        let mut words = line.split_whitespace();
        
        if let (Some(word1), Some(word2)) = (words.next(), words.next()) {
            merges.push((word1.into(), word2.into()));
        }
    }
    
    Ok(merges)
}

fn construct_vocab(chars: impl Iterator<Item=char> + Clone, merges: &[(String, String)]) -> Vec<String> {
    let iter = chars.map(String::from);
    let mut vocab: Vec<_> = iter.clone().chain( iter.map(|c| c + "</w>") ).collect();

    for merge in merges {
        vocab.push(format!("{}{}", merge.0, merge.1));
    }

    vocab.extend(["<|startoftext|>".to_string(), "<|endoftext|>".to_string()]);

    return vocab;
}

pub struct SimpleTokenizer {
    byte_encoder: HashMap<u8, char>,
    byte_decoder: HashMap<char, u8>,
    encoder: HashMap<String, u32>,
    decoder: HashMap<u32, String>,
    bpe_ranks: HashMap<(String, String), u32>,
    cache: HashMap<String, String>,
    pat: Regex, 
}

impl SimpleTokenizer {
    pub fn new() -> io::Result<Self> {
        let byte_unicode_values = bytes_to_unicode();

        let byte_encoder: HashMap<_, _> = byte_unicode_values.iter().cloned().collect();
        let byte_decoder = byte_encoder.iter().map(|(k,v)| (*v,*k)).collect();

        let merges = load_merges("bpe_simple_vocab_16e6.txt")?;
        let merges = merges[1..49152-256-2+1].to_vec();

        let vocab = construct_vocab(byte_unicode_values.into_iter().map(|(_, u)| u), &merges[..]);

        let encoder: HashMap<String, u32> = vocab.iter().cloned().zip((0..).into_iter()).collect();
        let decoder: HashMap<u32, String> = encoder.iter().map(|(k, v)| (*v, k.clone())).collect();
        let bpe_ranks = merges.iter().cloned().zip((0..).into_iter()).collect();
        let cache = HashMap::from([
            ("<|startoftext|>".to_string(), "<|startoftext|>".to_string()), 
            ("<|endoftext|>".to_string(), "<|endoftext|>".to_string()), 
        ]);

        let pat = Regex::new(r"(?i)<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+").unwrap();

        Ok( SimpleTokenizer {
            byte_encoder: byte_encoder,
            byte_decoder: byte_decoder,
            encoder: encoder,
            decoder: decoder,
            bpe_ranks: bpe_ranks,
            cache: cache,
            pat: pat, 
        } )
    }

    pub fn bpe(&self, token: &str) -> String {
        if let Some(word) = self.cache.get(token) {
            return word.clone();
        }
        
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        word.last_mut().map(|w| *w += "</w>");
        let mut pairs = get_pairs(&word);
        
        if pairs.is_empty() {
            return format!("{}{}", token, "</w>");
        }
        
        loop {
            let bigram = pairs.iter()
                .filter(|pair| self.bpe_ranks.contains_key(pair))
                .min_by_key(|&pair| self.bpe_ranks[pair]);

            if bigram.is_none() {
                break;
            }

            let (first, second) = bigram.unwrap();
            let mut new_word = Vec::new();
            let mut i = 0;
            while i < word.len() {
                if let Some( (j, _) ) = word.iter().enumerate().skip(i).find(|(_, w)| w == &first) {
                    new_word.extend(word[i..j].iter().cloned());
                    i = j;
                } else {
                    new_word.extend(word[i..].iter().cloned());
                    break;
                }
                
                if &word[i] == first && i < word.len() - 1 && &word[i + 1] == second {
                    new_word.push(format!("{}{}", first, second));
                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }
            
            word = new_word;
            if word.len() == 1 {
                break;
            } else {
                pairs = get_pairs(&word[..])
            }
        }

        let word = word.join(" ");
        //self.cache.insert(token.into(), word);
        return word;
    }
    
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let cleaned_text = whitespace_clean(text.trim()).to_lowercase();

        let mut bpe_tokens: Vec<u32> = Vec::new();

        for m in self.pat.find_iter(&cleaned_text) {
            let token = m.as_str();
            let token: String = token.as_bytes().into_iter().map(|b| self.byte_encoder[b]).collect();
            bpe_tokens.extend(self.bpe(&token).split(' ').map(|bpe_token| self.encoder[bpe_token]))
        }

        return bpe_tokens;
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        let text: String = tokens.iter().map(|t| self.decoder[t].as_str()).collect();
        let decoded_bytes: Vec<u8> = text.chars()
            .map(|c| self.byte_decoder[&c])
            .collect();

        String::from_utf8_lossy(&decoded_bytes[..]).replace("</w>", " ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let tokenizer = SimpleTokenizer::new().unwrap();

        let text = "Hello world! <|startoftext|>asdf<|startoftext|>";
        let target_encode = [3306, 1002, 256, 49406, 587, 10468, 49406];
        let target_decode = "hello world ! <|startoftext|>asdf <|startoftext|>"; // extra spaces sometimes

        let encoded = tokenizer.encode(&text);
        assert_eq!(&target_encode[..], &encoded[..]);
        let decoded = tokenizer.decode(&encoded[..]);
        assert_eq!(target_decode, decoded);
    }
}