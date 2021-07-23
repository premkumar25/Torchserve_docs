# Torchserve_docs
Document to serve a model using torchserve



Torchserve

TorchServe is a flexible and easy to use tool for serving PyTorch models.

Requirements:
For Conda python 3.8 is required
Openjdk version 11 is required

Installation:
For Conda,
conda install torchserve torch-model-archiver torch-workflow-archiver -c pytorch
For Pip
pip install torchserve torch-model-archiver torch-workflow-archiver

For Docker
Either we can build a docker image from github repo or directly pull the image from docker hub
git clone https://github.com/pytorch/serve.git
cd serve/docker
./build_image.sh  (For CPU based image)
./build_image.sh -g	(For GPU based image)
	Check the documentation for more information.

docker pull pytorch/torchserve		(From Docker hub)

Serve model

In order to serve the model we need to create a .mar(Model Archiver) file. 
There are two methods to create the .mar file.

Method 1:
Install torch-model-archiver
pip install torch-model-archiver
Create a model.py file and handler.py file
Model.py file contains the architecture of your model 

Below is the sample for mnist model



import torch
import torch.nn.functional as F


from torch import nn


class Net(nn.Module):


   def __init__(self):
       super(Net, self).__init__()
       self.conv1 = nn.Conv2d(1,32,3,1)
       self.conv2=nn.Conv2d(32,64,3,1)
       self.dropout2 = n.Dropout2d(0.5)
       self.fc1 = nn.Linear(9216, 128)
       self.fc2 = nn.Linear(128, 10)

   def forward(self, x):
       x = self.conv1(x)
       x = F.relu(x)
       x = self.conv2(x)
       x = F.max_pool2d(x, 2)
       x = torch.flatten(x, 1)
       x = self.fc1(x)
       x = F.relu(x)
       x = self.dropout2(x)
       x = self.fc2(x)
       output = F.log_softmax(x, dim=1)


       return output


Default handler is present in torchserve and we can build custom handler as per requirement
Run this command to create the .mar file

torch-model-archiver --model-name mnist --version 1.0 --model-file model.py --serialized-file mnist.pth --extra-files index_to_name.json --handler image_classifier
index_to_name.json : This file contains the mapping of predicted index to class. The default TorchServe handles returns the predicted index and probability. This file can be passed to model archiver using --extra-files parameter ({1:’one’, 2:’two’, 3:’three’})
Image_classifier is the default handler present in torchserve for image classification
Method 2:
serialized-file (.pt) : This file represents the state_dict in case of eager mode model or an executable ScriptModule in case of TorchScript.
index_to_name.json : This file contains the mapping of predicted index to class. The default TorchServe handles returns the predicted index and probability. This file can be passed to model archiver using --extra-files parameter
handler : TorchServe default handler's name or path to custom inference handler(.py)
torch-model-archiver --model-name <model_name> --version <model_version_number> --serialized-file <path_to_executable_script_module> --extra-files <path_to_index_to_name_json_file> --handler <path_to_custom_handler_or_default_handler_name>
This method doesn’t require model.py file as serialized file (.pt) is saved with the weights
Refer to this documentation for more information.
Once .mar file is created save it in a folder ‘model_store’
Once everything is in place, we can either run in from conda env or using docker.
	From Conda:
Run this command to start torchserve
torchserve --start --ncs --model-store model_store --models mnist.mar
model_store: path of the model_store
mnist.mar: name of mar file saved
Stop torchserve using this command
torchserve --stop
	
 
 
	From Docker:
Start the docker container from the folder where .mar is present
Run this command
sudo docker run --rm -d -it -p 80800:8080 -p 8081:8081 -v $(pwd):/home/model-server/model-store pytorch/torchserve:latest-gpu torchserve --start --model-store model-store --models resnet=mnist.mar

Inference can be done by both CURL and gRPC
	
CURL:
curl http://127.0.0.1:8081/models
Lists the models currently running.
Inference can be obtained by 
curl http://127.0.0.1:8080/predictions/<model_name> -T <inference_file>

gRPC:
Install grpc python dependencies :
pip install -U grpcio protobuf grpcio-tools
Clone github repo
git clone https://github.com/pytorch/serve.git
cd serve
Generate inference client using proto files
python -m grpc_tools.protoc --proto_path=frontend/server/src/main/resources/proto/ --python_out=ts_scripts --grpc_python_out=ts_scripts frontend/server/src/main/resources/proto/inference.proto frontend/server/src/main/resources/proto/management.proto
Run inference using a sample client gRPC python client
python ts_scripts/torchserve_grpc_client.py infer <model_name> <inference_file>

