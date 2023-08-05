use bincode;
use burn::record::{PrecisionSettings, Recorder, RecorderError, FileRecorder};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use std::marker::PhantomData;
use serde::{de::DeserializeOwned, Serialize};
//use super::{bin_config, PrecisionSettings, Recorder, RecorderError};

fn bin_config() -> bincode::config::Configuration {
    bincode::config::standard()
}

macro_rules! str2reader {
    ($file:expr) => {{
        $file.set_extension(<Self as FileRecorder>::file_extension());
        let path = $file.as_path();
        File::open(path).map_err(|err| match err.kind() {
            std::io::ErrorKind::NotFound => RecorderError::FileNotFound(err.to_string()),
            _ => RecorderError::Unknown(err.to_string()),
        }).map(|file| BufReader::new(file)) // wrap File in BufReader
    }};
}

macro_rules! str2writer {
    ($file:expr) => {{
        $file.set_extension(<Self as FileRecorder>::file_extension());
        let path = $file.as_path();

        if path.exists() {
            //log::info!("File exists, replacing");
            std::fs::remove_file(path).map_err(|err| RecorderError::Unknown(err.to_string()))?;
        }

        File::create(path).map_err(|err| match err.kind() {
            std::io::ErrorKind::NotFound => RecorderError::FileNotFound(err.to_string()),
            _ => RecorderError::Unknown(err.to_string()),
        }).map(|file| BufWriter::new(file))  // wrap File in BufWriter
    }};
}

#[derive(Debug, Default, Clone)]
pub struct BinFileRecorderBuffered<S: PrecisionSettings> {
    _settings: PhantomData<S>,
}

impl<S: PrecisionSettings> BinFileRecorderBuffered<S> {
    pub fn new() -> Self {
        BinFileRecorderBuffered {
            _settings: PhantomData, 
        }
    }
}

impl<S: PrecisionSettings> FileRecorder for BinFileRecorderBuffered<S> {
    fn file_extension() -> &'static str {
        "bin"
    }
}

impl<S: PrecisionSettings> Recorder for BinFileRecorderBuffered<S> {
    type Settings = S;
    type RecordArgs = PathBuf;
    type RecordOutput = ();
    type LoadArgs = PathBuf;

    fn save_item<I: Serialize>(
        &self,
        item: I,
        mut file: Self::RecordArgs,
    ) -> Result<(), RecorderError> {
        let config = bin_config();
        let mut writer = str2writer!(file)?;
        bincode::serde::encode_into_std_write(&item, &mut writer, config)
            .map_err(|err| RecorderError::Unknown(err.to_string()))?;
        Ok(())
    }

    fn load_item<I: DeserializeOwned>(&self, mut file: Self::LoadArgs) -> Result<I, RecorderError> {
        let mut reader = str2reader!(file)?;
        let state =
            bincode::serde::decode_from_std_read(&mut reader, bin_config())
                .map_err(|err| RecorderError::Unknown(err.to_string()))?;
        Ok(state)
    }
}
