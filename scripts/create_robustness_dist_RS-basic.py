import logging

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

from pathlib import Path

import torch

torch.manual_seed(0)
import torchvision  # noqa: E402
import torchvision.transforms as transforms

from robustness_experiment_box.database.dataset.experiment_dataset import ExperimentDataset
from robustness_experiment_box.database.dataset.pytorch_experiment_dataset import PytorchExperimentDataset
from robustness_experiment_box.database.experiment_repository import ExperimentRepository
from robustness_experiment_box.dataset_sampler.dataset_sampler import DatasetSampler
from robustness_experiment_box.dataset_sampler.predictions_based_sampler import PredictionsBasedSampler
from robustness_experiment_box.epsilon_value_estimator.binary_search_epsilon_value_estimator import (
    BinarySearchEpsilonValueEstimator,
)
from robustness_experiment_box.epsilon_value_estimator.epsilon_value_estimator import EpsilonValueEstimator
from robustness_experiment_box.verification_module.certification import RandomizedSmoothingCertifier
from robustness_experiment_box.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)
from robustness_experiment_box.verification_module.property_generator.property_generator import PropertyGenerator


def create_distribution(
    experiment_repository: ExperimentRepository,
    dataset: ExperimentDataset,
    dataset_sampler: DatasetSampler,
    epsilon_value_estimator: EpsilonValueEstimator,
    property_generator: PropertyGenerator,
):
    network_list = experiment_repository.get_network_list()
    failed_networks = []

    networks = experiment_repository.get_network_list()
    
    for network in networks:
        try:
            sampled_data = dataset_sampler.sample(network, dataset)
            for data_point in sampled_data:
                context = experiment_repository.create_verification_context(
                    network, 
                    data_point,
                    property_generator
                )
                
                result = epsilon_value_estimator.compute_epsilon_value(context)
                experiment_repository.save_result(result)
                
        except Exception as e:
            logging.error(f"Failed on {network}: {str(e)}")
            continue

    experiment_repository.save_plots()
    logging.info(f"Failed for networks: {failed_networks}")


#RS Paramerters
RS_CONFIG = {
    "sigma": 0.5,
    "alpha": 0.01,
    "n0": 1000,
    "n": 100000,
    "batch_size": 512
}

# DDS_CONFIG = {
#     "denoiser_path": "pretrained/mnist_diffusion.pt",
#     **RS_CONFIG
# }

def main():
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    
    # Experiment setup
    experiment_name = "RS_MNIST"
    experiment_repository_path = Path("../tests/test_experiment")
    network_folder = Path("data/MNIST/raw/models") #only mnist-net_256x4 for dev purpose
    
    torch_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transforms.ToTensor() # only approx 10k images
)

    dataset = PytorchExperimentDataset(dataset=torch_dataset)

    # Initialize repository
    experiment_repository = ExperimentRepository(base_path=experiment_repository_path, network_folder=network_folder)
    property_generator = One2AnyPropertyGenerator()
    
    # Certification module
    rs_module = RandomizedSmoothingCertifier(**RS_CONFIG)
    #dds_module = DenoisedSmoothingModule(**DDS_CONFIG)

    # Certification pipeline
    epsilon_value_estimator  = BinarySearchEpsilonValueEstimator(
        epsilon_value_list=[0.0],  # Not used for RS
        verifier=rs_module  # Can switch to dds_module
    )
    dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=True)
    experiment_repository.initialize_new_experiment(experiment_name)

    experiment_repository.save_configuration(
        dict(
            experiment_name=experiment_name,
            experiment_repository_path=str(experiment_repository_path),
            network_folder=str(network_folder),
            dataset=str(dataset)
        ))
    
    create_distribution(experiment_repository, dataset, dataset_sampler, epsilon_value_estimator, property_generator)
    


if __name__ == "__main__":
    main()
