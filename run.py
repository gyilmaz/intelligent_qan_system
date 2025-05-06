# run.py
import hydra
from omegaconf import DictConfig, OmegaConf
from src import data_processing, vector_db, api # Import your module

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg)) # Optional: print config

    # Example of running a specific pipeline step
    if cfg.pipeline == "process_data":
        print("Running data processing pipeline...")
        processed_data = data_processing.process_handbook_pdf(cfg) # Ensure you call the PDF processing function
        # Now you have your chunks in the 'processed_data' list

    elif cfg.pipeline == "build_index":
         print("Running index building pipeline...")
         vector_db.build_vector_index(cfg) # Call the new index building function

    elif cfg.pipeline == "run_api":
         print("Initializing API configuration and models...")
         api.initialize_api_config_and_models(cfg) # Pass the fully resolved Hydra config
         print("Starting API server...")
         api.run_api(cfg) # This function now just starts Uvicorn

    elif cfg.pipeline == "run_ui":
         print("Running UI...")
         # Call UI runner function (will create later)

    else:
        print("Please specify a pipeline step, e.g., 'python run.py pipeline=process_data'")


if __name__ == "__main__":
    main()