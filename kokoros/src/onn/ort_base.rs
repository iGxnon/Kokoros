#[cfg(feature = "cuda")]
use ort::execution_providers::cuda::CUDAExecutionProvider;
#[cfg(feature = "coreml")]
use ort::execution_providers::coreml::CoreMLExecutionProvider;
use ort::execution_providers::cpu::CPUExecutionProvider;
use ort::session::builder::SessionBuilder;
use ort::session::Session;

pub trait OrtBase {
    fn load_model(&mut self, model_path: String) -> Result<(), String> {
        #[cfg(feature = "cuda")]
        let providers = [CUDAExecutionProvider::default().build()];

        #[cfg(feature = "coreml")]
        let providers = [
            CoreMLExecutionProvider::default().build(),
            CPUExecutionProvider::default().build()
        ];

        #[cfg(all(not(feature = "cuda"), not(feature = "coreml")))]
        let providers = [CPUExecutionProvider::default().build()];

        match SessionBuilder::new() {
            Ok(builder) => {
                let session = builder
                    .with_execution_providers(providers)
                    .map_err(|e| format!("Failed to build session: {}", e))?
                    .commit_from_file(model_path)
                    .map_err(|e| format!("Failed to commit from file: {}", e))?;
                self.set_sess(session);
                Ok(())
            }
            Err(e) => Err(format!("Failed to create session builder: {}", e)),
        }
    }

    fn set_sess(&mut self, sess: Session);
    fn sess(&self) -> Option<&Session>;
}
