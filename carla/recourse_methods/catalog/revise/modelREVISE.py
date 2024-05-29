from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from carla import log
from carla.data.api import Data
from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.autoencoder import VariationalAutoencoder
from carla.recourse_methods.processing.counterfactuals import (merge_default_parameters)

class Revise(RecourseMethod):
    """
    Implementation of Revise from Joshi et.al. [1]_.

    Parameters
    ----------
    mlmodel : carla.model.MLModel
        Black-Box-Model
    data: carla.data.Data
        Dataset to perform on
    hyperparams : dict
        Dictionary containing hyperparameters. See notes below for its contents.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "data_name": str
            name of the dataset
        * "lambda": float, default: 0.5
            Decides how similar the counterfactual is to the factual
        * "optimizer": {"adam", "rmsprop"}
            Optimizer for generation of counterfactuals.
        * "lr": float, default: 0.1
            Learning rate for Revise.
        * "max_iter": int, default: 1000
            Number of iterations for Revise optimization.
        * "target_class": int, default: [1]
            List of one-hot-encoded target class.
        * "binary_cat_features": bool, default: True
            If true, the encoding of x is done by drop_if_binary.
        * "vae_params": Dict
            With parameter for VAE.

            + "layers": list
                Number of neurons and layer of autoencoder.
            + "train": bool
                Decides if a new autoencoder will be learned.
            + "lambda_reg": flot
                Hyperparameter for variational autoencoder.
            + "epochs": int
                Number of epochs to train VAE
            + "lr": float
                Learning rate for VAE training
            + "batch_size": int
                Batch-size for VAE training

    .. [1] Shalmali Joshi, Oluwasanmi Koyejo, Warut Vijitbenjaronk, Been Kim, and Joydeep Ghosh.2019.
            Towards Realistic  Individual Recourse  and Actionable Explanations  in Black-BoxDecision Making Systems.
            arXiv preprint arXiv:1907.09615(2019).
    """

    _DEFAULT_HYPERPARAMS = {
        "data_name": None,
        "lambdas": [1, 0.5, 1], # the proximity lambda and the lambda for the process constraints 
        "optimizer": "adam", #adam, or RMSprop (optimizer for the generation of counterfactuals)
        "lr": 0.1, # learning rate for Revise
        "max_iter": 2000, #number of iterations for Revise optimization
        "target_class": [1], # target class
        "vocab_size": None, #vocab size
        "max_prefix_length":None, #the maximum sequence length
        "threshold": 0.5, # the threshold of probability before you consider the predicted to be flipped
        "loss_diff_threshold": 1e-5, # the loss difference threshold (0.00001), 
                                 # if less than threshold, a stopping threshold is triggered
        "vae_params": { # the vae params is a dictionary with the learning parameters for the VAE
            "layers": None, # hidden size, latent size, lstm size of the VAE
            "train": True,  # force VAE training or not
            "joint_constraint" : True,
            "lambda_reg": 1e-6, # this is the lambda for the optimizer of the VAE
            "epochs": 5, # the epochs for VAE training
            "lr": 1e-3, # learning rate optimizer VAE
            "batch_size": 128, # batch size VAE
            "dropout": 0 # the dropout for the LSTM VAE
        },
    }

    def __init__(self, mlmodel: MLModel, dataset: Data, hyperparams: Dict = None, constraintmodel = None) -> None:

        supported_backends = ["pytorch"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )

        super().__init__(mlmodel)
        self._params = merge_default_parameters(hyperparams, self._DEFAULT_HYPERPARAMS)
        self.constraint_violation = constraintmodel
        self.data_train = dataset.df_train
        self.hinge_loss_weight = self._params["lambdas"][0]
        self.proximity_weight = self._params["lambdas"][1]
        self.constraint_penalty = self._params["lambdas"][2]
        self._optimizer = self._params["optimizer"]
        self._lr = self._params["lr"]
        self._max_iter = self._params["max_iter"]
        self._target_class = self._params["target_class"]
        self.vocab_size = self._params['vocab_size']
        self._max_prefix_length = self._params['max_prefix_length']
        self.stopping_threshold = self._params["threshold"]
        self.loss_diff_thres = self._params["loss_diff_threshold"]     
        self.beta = 0.1
        self.verbose = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        vae_params = self._params["vae_params"]
        self.joint_constraint_in_loss = vae_params["joint_constraint"]
        self.vae = VariationalAutoencoder(
            self._params["data_name"], vae_params["layers"], self.vocab_size, self._max_prefix_length, vae_params["epochs"],
            self.constraint_violation, joint_constraint_in_loss= self.joint_constraint_in_loss
        )

        if vae_params["train"]:
            self.vae.fit(
                xtrain=self.data_train,
                lambda_reg=vae_params["lambda_reg"],
                epochs=vae_params["epochs"],
                lr=vae_params["lr"],
                batch_size=vae_params["batch_size"]
            )
        
        else:
            try:
                self.vae.load()
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "Loading of Autoencoder failed. {}".format(str(exc))
                )
    def get_counterfactuals(self, factuals):
        # This stores the generated counterfactuals
        list_cfs = self._counterfactual_optimization(factuals)
        # this checks whether it actually is a counterfactual. For now, we will assume that this is the case when the probability for 1 is higher than probability for 0.
        #cf_df = check_counterfactuals(self._mlmodel, list_cfs, negative_label=0)
        return list_cfs
    
    def are_tensors_identical(self, cf, tensor_list):
        reference_tensor =cf
        for tensor in tensor_list:
            if torch.equal(reference_tensor, tensor):
                return False
        return True
    
    def add_candidate_counterfactual(self, cf, loss):
        cf = cf.clone().detach()
        argmax = torch.argmax(cf.detach(), dim=2)
        argmax = argmax[0].numpy()
        softmax_cf = F.softmax(cf, dim=2)        
        candidate_counterfactual = softmax_cf.cpu().detach().numpy()
        #one_hot_encoded = np.zeros_like(candidate_counterfactual) # Create one-hot encoded values
        # Fill the one-hot encoding matrix with the label values
        #for i in range(len(argmax)):
        #    one_hot_encoded[0, i, argmax[i]] = 1

        # Print the resulting one-hot encoding matrix
        if not any(np.array_equal(argmax, array) for array in self.candidate_counterfactuals_argmax):
            self.candidate_counterfactuals_argmax.append(argmax) #this is to check whether the argmaxed version is not already used
            
            self.candidate_counterfactuals.append(candidate_counterfactual)
            self.candidate_distances.append(loss.cpu().detach().numpy())
            if self.verbose:
                print('new cf added')

    def mask_out_tensor(self, tensor):
        # Find the index of the maximum value (EoS token) in each tensor
        _, index = torch.max(tensor, dim=2)
        index = index
        result_indexes = []
        for row in index:
            index2 = (row == self.vocab_size-1).nonzero(as_tuple=False)
            if len(index2) > 0:
                result_indexes.append(index2[0, 0].item())
            else:
                result_indexes.append(-1)   # result contains the indexes of where the value 

        one_hot_masked = tensor.clone()
        for idx in range(one_hot_masked.shape[0]):
            if result_indexes[idx] == -1:
                continue
            else:
                for j in range(result_indexes[idx]+1, one_hot_masked.shape[1]):
                    one_hot_masked[idx][j,:] = torch.tensor([0]*one_hot_masked.shape[2])

        return one_hot_masked

    def _counterfactual_optimization(self, torch_fact):
        # This function is responsible for the optimization process to generate counterfactuals
        # Dataloader to prepare data for optimization steps. We take a batch size of 1 because we generate a counterfactual for each trace seperately.
        test_loader = torch.utils.data.DataLoader(
            torch_fact, batch_size=1, shuffle=False
        )

        list_cfs = [] #this stores the counterfactuals
        for query_instance in test_loader:
            self.query_instance = query_instance
            
            violations, constraints = self.constraint_violation.count_violations(torch.tensor(query_instance.clone().detach(), dtype=torch.float32).squeeze(), constraint_list = 'total')
            print('original query:', np.argmax(self.query_instance.clone().detach().numpy(), axis=2))
            print('the constraints are:', constraints)
            print('the original query has', (violations/len(constraints))*100, '%/ of constraint violations')
            
            target = torch.FloatTensor(self._target_class).to(self.device) #ensure both target and input are on the same device and both are a floattensor
            z = self.vae.encode(query_instance.float())[0]  # encode the query instance
            #z = self.vae._reparametrization_trick(mu, logvar)
            z = z.clone().detach().requires_grad_(True)  #the original REVISE algorithm only takes the mean of the latent variable

            if self._optimizer == "adam":
                optim = torch.optim.Adam([z], self._lr)
            else:
                optim = torch.optim.RMSprop([z], self._lr)

            self.candidate_counterfactuals = []  # all possible counterfactuals
            self.candidate_counterfactuals_argmax = []
            self.candidate_distances = [] # distance of the possible counterfactuals from the intial value 
            cf_list_check = []
            # Inside your _counterfactual_optimization function
            for idx in range(self._max_iter):
                cf = self.vae.decode(z) 
                cf = self.mask_out_tensor(cf)
                cf_list_check.append(cf[0])
                output_orig = self._mlmodel.predict_proba(query_instance)[0]
                output = self._mlmodel.predict_proba(cf)[0]
                z.requires_grad = True
                if self.verbose:
                    print('the current cf:', np.argmax(cf.clone().detach().numpy(), axis=2))
                loss = self.compute_loss(cf, query_instance, target)
                loss = loss.to(self.device)
                # Set the VAE model to evaluation mode before backward pass
                self.vae.eval()
                if ((self._target_class[0] == 0 and output.item() < self.stopping_threshold) or
                (self._target_class[0] == 1 and output.item() > self.stopping_threshold)):
                    self.add_candidate_counterfactual(cf, loss)  #we add the counterfactual if it actually is a counterfactual
                #print('counterfactual', torch.argmax(cf, dim=2))
                loss.backward() # After loss.backward(), check the gradients of z
                
                if self.verbose:
                    print('are the CF unique?', self.are_tensors_identical(cf[0],cf_list_check))
                    print('original', torch.argmax(query_instance, dim=2))
                    print('the probability for the original', output_orig)
                    print('the predicted prob counterfactual', output.item())
                    print("Gradients of z:")
                    print(z.grad is not None)
                
                optim.step()
                optim.zero_grad()  # Clear gradients for the next iteration
                cf.detach_()

            # print out the counterfactuals
            if len(self.candidate_counterfactuals):
                log.info("Counterfactual found!")
                for i in self.candidate_counterfactuals:
                    ax_indices = np.argmax(i, axis=2)
                    print('the counterfactual', ax_indices) 
                    violations, _ = self.constraint_violation.count_violations(torch.tensor(i).squeeze(), constraint_list = 'total')
                    print('process constraints?', (violations/len(constraints))*100, '%/ of constraint violations')
                    list_cfs.append(i)
            else:
                log.info("No counterfactual found")
                list_cfs.append(query_instance.cpu().detach().numpy())
        tensor_cfs = np.concatenate(np.array(list_cfs),axis=0)
        return tensor_cfs

    def compute_loss(self, cf_initialize, query_instance, target):
        query_instance = query_instance.cpu()
        # Computes the first component of the loss function (a differentiable hinge loss function)
        epsilon = 1e-8  # Small epsilon to avoid division by zero
        probability = self._mlmodel.predict_proba(cf_initialize)[0]
        #print('probability', probability)
        temp_logits = torch.log(probability + epsilon) - torch.log(1 - probability + epsilon)
        all_ones = torch.ones_like(target)
        labels = 2 * target - all_ones
        loss1 = torch.log(1 + torch.exp(-torch.mul(labels, temp_logits)))  
      
        # Compute weighted distance between two vectors.
        softmax_cf = F.softmax(cf_initialize, dim=2).cpu()
        delta = abs(softmax_cf - query_instance)
        l1_loss =  torch.sum(torch.abs(delta))
        l2_loss = torch.sqrt(torch.sum(delta ** 2))
        loss2 = self.beta*l1_loss + l2_loss
        
        # compute the constraint violations
        violations, _ = self.constraint_violation.count_violations(cf_initialize.detach().squeeze(), constraint_list = 'total')

        if self.verbose:
            print('loss1', self.hinge_loss_weight * loss1)
            print('loss2', self.proximity_weight * loss2)
            #print('loss3', self.constraint_penalty * violations)
        
        if self.joint_constraint_in_loss:
            total_loss =  self.hinge_loss_weight*loss1 + self.proximity_weight * loss2 + self.constraint_penalty * violations

        elif self.joint_constraint_in_loss==False:
            total_loss =  self.hinge_loss_weight*loss1 + self.proximity_weight * loss2
 
        if total_loss <0:
             print('negative loss so you should change the weights')
            
        return total_loss
