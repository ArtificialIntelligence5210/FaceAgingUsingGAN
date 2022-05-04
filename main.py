from argparse import ArgumentParser    #We have used argparse which provides CLI flexibility and Argumentparser library parses command line arguments and gives a useful message

import yaml                    #imported yaml for configuration files
from pytorch_lightning import Trainer   #Importing Trainer from pytorch which automates everything

from gan_module import AgingGAN #imported AgingGan library which is a DeepLearning model

parser = ArgumentParser()  #it holds all the information necessary to parse the CLI into the python data types

#We can fill the information about the program arguments in ArgumentPaser by calling the add_argument
parser.add_argument('--config', default='configs/aging_gan.yaml', help='Config to use for training') # 

# we have defined the main function below
def main():
    args = parser.parse_args()     #parses arguments through parse_args()
    with open(args.config) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)  # To convert YAML file to python we can use yaml.load function and given Loader=FullLoader which loads the full YAML language and avoid security flaws
        
    print(config)            #print the configuration
    model = AgingGAN(config)           #Storing AgingGAN(config) in model
    trainer = Trainer(max_epochs=config['epochs'], gpus=config['gpus'], auto_scale_batch_size='binsearch')            #max_epochs stop training once the the number of epochs is reached,gpu = Number of GPUs to train on, auto_scale_batch_size is to find the large batch size that fits into memory
    trainer.fit(model)   #fitting your model


if __name__ == '__main__':     #This condition will execute only when program is executed by python interpreter
    main()
