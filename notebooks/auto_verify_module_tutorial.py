from datetime import datetime
from pathlib import Path


from auto-verify.verifier import Nnenum

from robustness_experiment_box.analysis.report_creator import ReportCreator
from robustness_experiment_box.database.dataset.image_file_dataset import ImageFileDataset
from robustness_experiment_box.database.dataset.pytorch_experiment_dataset import PytorchExperimentDataset
from robustness_experiment_box.database.network import Network
from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.dataset_sampler.predictions_based_sampler import PredictionsBasedSampler
from robustness_experiment_box.epsilon_value_estimator.binary_search_epsilon_value_estimator import (
    BinarySearchEpsilonValueEstimator,
)
from robustness_experiment_box.verification_module.auto_verify_module import AutoVerifyModule
from robustness_experiment_box.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)


import torch

torch.manual_seed(0)
import pandas as pd
import torchvision
import torchvision.transforms as transforms




# define pytorch dataset. Preprocessing can be defined in the transform parameter
torch_dataset = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())

# wrap pytorch dataset into experiment dataset to keep track of image id
experiment_dataset = PytorchExperimentDataset(dataset=torch_dataset)

# work on subset of the dataset to keep experiment small
experiment_dataset = experiment_dataset.get_subset([x for x in range(0, 10)])



# Alternatively, one can also use a custom dataset from the storage. For this, one can make use of the ImageFileDataset class

# Here, one can also add a preprocessing. However, as of now just the loading of torch tensors from the directory is supported
preprocessing = transform = torchvision.transforms.Compose([torchvision.transforms.Normalize((0.1307,), (0.3081,))])
custom_experiment_dataset = ImageFileDataset(
    image_folder=Path("../tests/test_experiment/data/images"),
    label_file=Path("../tests/test_experiment/data/image_labels.csv"),
    preprocessing=preprocessing,
)