import numpy as np
import pandas as pd
import torch
from sklearn.mixture import BayesianGaussianMixture


class DataTransformer():
    
    """
    Transformer class responsible for processing data to train the CTABGANSynthesizer model
    
    Variables:
    1) train_data -> input dataframe 
    2) categorical_list -> list of categorical columns
    3) mixed_dict -> dictionary of mixed columns
    4) n_clusters -> number of modes to fit bayesian gaussian mixture (bgm) model
    5) eps -> threshold for ignoring less prominent modes in the mixture model 
    6) ordering -> stores original ordering for modes of numeric columns
    7) output_info -> stores dimension and output activations of columns (i.e., tanh for numeric, softmax for categorical)
    8) output_dim -> stores the final column width of the transformed data
    9) components -> stores the valid modes used by numeric columns
    10) filter_arr -> stores valid indices of continuous component in mixed columns
    11) meta -> stores column information corresponding to different data types i.e., categorical/mixed/numerical


    Methods:
    1) __init__() -> initializes transformer object and computes meta information of columns
    2) get_metadata() -> builds an inventory of individual columns and stores their relevant properties
    3) fit() -> fits the required bgm models to process the input data
    4) transform() -> executes the transformation required to train the model
    5) inverse_transform() -> executes the reverse transformation on data generated from the model
    
    """    
    
    def __init__(self, train_data=pd.DataFrame, categorical_list=[], mixed_dict={}, n_clusters=10, eps=0.005):
        
        self.meta = None
        self.train_data = train_data
        self.categorical_columns= categorical_list
        self.mixed_columns= mixed_dict
        self.n_clusters = n_clusters
        self.eps = eps
        self.ordering = []
        self.output_info = []
        self.output_dim = 0
        self.components = []
        self.filter_arr = []
        self.meta = self.get_metadata()
        
    def get_metadata(self):
        
        meta = []
    
        for index in range(self.train_data.shape[1]):
            column = self.train_data.iloc[:,index]
            if index in self.categorical_columns:
                mapper = column.value_counts().index.tolist()
                meta.append({
                        "name": index,
                        "type": "categorical",
                        "size": len(mapper),
                        "i2s": mapper
                })
            elif index in self.mixed_columns.keys():
                meta.append({
                    "name": index,
                    "type": "mixed",
                    "min": column.min(),
                    "max": column.max(),
                    "modal": self.mixed_columns[index]
                })
            else:
                meta.append({
                    "name": index,
                    "type": "continuous",
                    "min": column.min(),
                    "max": column.max(),
                })            

        return meta

    def fit(self):
        
        data = self.train_data.values
        
        # stores the corresponding bgm models for processing numeric data
        model = []
        
        # iterating through column information
        for id_, info in enumerate(self.meta):
            if info['type'] == "continuous":
                # fitting bgm model  
                gm = BayesianGaussianMixture(
                    self.n_clusters, 
                    weight_concentration_prior_type='dirichlet_process',
                    weight_concentration_prior=0.001, # lower values result in lesser modes being active
                    max_iter=100,n_init=1, random_state=42)
                gm.fit(data[:, id_].reshape([-1, 1]))
                model.append(gm)
                # keeping only relevant modes that have higher weight than eps and are used to fit the data
                old_comp = gm.weights_ > self.eps
                mode_freq = (pd.Series(gm.predict(data[:, id_].reshape([-1, 1]))).value_counts().keys())
                comp = []
                for i in range(self.n_clusters):
                    if (i in (mode_freq)) & old_comp[i]:
                        comp.append(True)
                    else:
                        comp.append(False)
                self.components.append(comp) 
                self.output_info += [(1, 'tanh'), (np.sum(comp), 'softmax')]
                self.output_dim += 1 + np.sum(comp)
                
            elif info['type'] == "mixed":
                
                # in case of mixed columns, two bgm models are used
                gm1 = BayesianGaussianMixture(
                    self.n_clusters, 
                    weight_concentration_prior_type='dirichlet_process',
                    weight_concentration_prior=0.001, max_iter=100,
                    n_init=1,random_state=42)
                gm2 = BayesianGaussianMixture(
                    self.n_clusters,
                    weight_concentration_prior_type='dirichlet_process',
                    weight_concentration_prior=0.001, max_iter=100,
                    n_init=1,random_state=42)
                
                # first bgm model is fit to the entire data only for the purposes of obtaining a normalized value of any particular categorical mode
                gm1.fit(data[:, id_].reshape([-1, 1]))
                
                # main bgm model used to fit the continuous component and serves the same purpose as with purely numeric columns
                filter_arr = []
                for element in data[:, id_]:
                    if element not in info['modal']:
                        filter_arr.append(True)
                    else:
                        filter_arr.append(False)
                self.filter_arr.append(filter_arr)
                
                gm2.fit(data[:, id_][filter_arr].reshape([-1, 1]))
                
                model.append((gm1,gm2))
                
                # similarly keeping only relevant modes with higher weight than eps and are used to fit strictly continuous data 
                old_comp = gm2.weights_ > self.eps
                mode_freq = (pd.Series(gm2.predict(data[:, id_][filter_arr].reshape([-1, 1]))).value_counts().keys())  
                comp = []
                  
                for i in range(self.n_clusters):
                    if (i in (mode_freq)) & old_comp[i]:
                        comp.append(True)
                    else:
                        comp.append(False)

                self.components.append(comp)
                
                # modes of the categorical component are appended to modes produced by the main bgm model
                self.output_info += [(1, 'tanh'), (np.sum(comp) + len(info['modal']), 'softmax')]
                self.output_dim += 1 + np.sum(comp) + len(info['modal'])
            
            else:
                # in case of categorical columns, bgm model is ignored
                model.append(None)
                self.components.append(None)
                self.output_info += [(info['size'], 'softmax')]
                self.output_dim += info['size']
        
        self.model = model

    def transform(self, data):
        
        # stores the transformed values
        values = []

        # used for accessing filter_arr for transforming mixed columns
        mixed_counter = 0
        
        # iterating through column information
        for id_, info in enumerate(self.meta):
            current = data[:, id_]
            if info['type'] == "continuous":
                # mode-specific normalization occurs here
                current = current.reshape([-1, 1])
                # means and stds of the modes are obtained from the corresponding fitted bgm model
                means = self.model[id_].means_.reshape((1, self.n_clusters))
                stds = np.sqrt(self.model[id_].covariances_).reshape((1, self.n_clusters))
                # values are then normalized and stored for all modes
                features = np.empty(shape=(len(current),self.n_clusters))
                # note 4 is a multiplier to ensure values lie between -1 to 1 but this is not always guaranteed
                features = (current - means) / (4 * stds) 

                # number of distict modes
                n_opts = sum(self.components[id_])                
                # storing the mode for each data point by sampling from the probability mass distribution across all modes based on fitted bgm model 
                opt_sel = np.zeros(len(data), dtype='int')
                probs = self.model[id_].predict_proba(current.reshape([-1, 1]))
                probs = probs[:, self.components[id_]]
                for i in range(len(data)):
                    pp = probs[i] + 1e-6
                    pp = pp / sum(pp)
                    opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)
                
                # creating a one-hot-encoding for the corresponding selected modes
                probs_onehot = np.zeros_like(probs)
                probs_onehot[np.arange(len(probs)), opt_sel] = 1

                # obtaining the normalized values based on the appropriately selected mode and clipping to ensure values are within (-1,1)
                idx = np.arange((len(features)))
                features = features[:, self.components[id_]]
                features = features[idx, opt_sel].reshape([-1, 1])
                features = np.clip(features, -.99, .99) 
                
                # re-ordering the one-hot-encoding of modes in descending order as per their frequency of being selected
                re_ordered_phot = np.zeros_like(probs_onehot)  
                col_sums = probs_onehot.sum(axis=0)
                n = probs_onehot.shape[1]
                largest_indices = np.argsort(-1*col_sums)[:n]
                for id,val in enumerate(largest_indices):
                    re_ordered_phot[:,id] = probs_onehot[:,val]
                
                # storing the original ordering for invoking inverse transform
                self.ordering.append(largest_indices)
                
                # storing transformed numeric column represented as normalized values and corresponding modes 
                values += [features, re_ordered_phot]
                  
            elif info['type'] == "mixed":
                
                # means and standard deviation of modes obtained from the first fitted bgm model
                means_0 = self.model[id_][0].means_.reshape([-1])
                stds_0 = np.sqrt(self.model[id_][0].covariances_).reshape([-1])

                # list to store relevant bgm modes for categorical components
                zero_std_list = []
                
                # means and stds needed to normalize relevant categorical components
                means_needed = []
                stds_needed = []

                # obtaining the closest bgm mode to the categorical component
                for mode in info['modal']:
                    # skipped for mode representing missing values
                    if mode!=-9999999:
                        dist = []
                        for idx,val in enumerate(list(means_0.flatten())):
                            dist.append(abs(mode-val))
                        index_min = np.argmin(np.array(dist))
                        zero_std_list.append(index_min)
                    else: continue

                
                # stores the appropriate normalized value of categorical modes
                mode_vals = []
                
                # based on the means and stds of the chosen modes for categorical components, their respective values are similarly normalized
                for idx in zero_std_list:
                    means_needed.append(means_0[idx])
                    stds_needed.append(stds_0[idx])
               
                for i,j,k in zip(info['modal'],means_needed,stds_needed):
                    this_val  = np.clip(((i - j) / (4*k)), -.99, .99) 
                    mode_vals.append(this_val)
                
                # for categorical modes representing missing values, the normalized value associated is simply 0
                if -9999999 in info["modal"]:
                    mode_vals.append(0)
                
                # transforming continuous component of mixed columns similar to purely numeric columns using second fitted bgm model
                current = current.reshape([-1, 1])
                filter_arr = self.filter_arr[mixed_counter]
                current = current[filter_arr]
    
                means = self.model[id_][1].means_.reshape((1, self.n_clusters))
                stds = np.sqrt(self.model[id_][1].covariances_).reshape((1, self.n_clusters))
                
                features = np.empty(shape=(len(current),self.n_clusters))
                features = (current - means) / (4 * stds)
                
                n_opts = sum(self.components[id_]) 
                probs = self.model[id_][1].predict_proba(current.reshape([-1, 1]))
                probs = probs[:, self.components[id_]]
                
                opt_sel = np.zeros(len(current), dtype='int')
                for i in range(len(current)):
                    pp = probs[i] + 1e-6
                    pp = pp / sum(pp)
                    opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)
                
                idx = np.arange((len(features)))
                features = features[:, self.components[id_]]
                features = features[idx, opt_sel].reshape([-1, 1])
                features = np.clip(features, -.99, .99)
                
                probs_onehot = np.zeros_like(probs)
                probs_onehot[np.arange(len(probs)), opt_sel] = 1
                
                # additional modes are appended to represent categorical component
                extra_bits = np.zeros([len(current), len(info['modal'])])
                temp_probs_onehot = np.concatenate([extra_bits,probs_onehot], axis = 1)
                
                # storing the final normalized value and one-hot-encoding of selected modes
                final = np.zeros([len(data), 1 + probs_onehot.shape[1] + len(info['modal'])])

                # iterates through only the continuous component
                features_curser = 0

                for idx, val in enumerate(data[:, id_]):
                    
                    if val in info['modal']:
                        # dealing with the modes of categorical component
                        category_ = list(map(info['modal'].index, [val]))[0]
                        final[idx, 0] = mode_vals[category_]
                        final[idx, (category_+1)] = 1
                    
                    else:
                        # dealing with the modes of continuous component
                        final[idx, 0] = features[features_curser]
                        final[idx, (1+len(info['modal'])):] = temp_probs_onehot[features_curser][len(info['modal']):]
                        features_curser = features_curser + 1

                # re-ordering the one-hot-encoding of modes in descending order as per their frequency of being selected
                just_onehot = final[:,1:]
                re_ordered_jhot= np.zeros_like(just_onehot)
                n = just_onehot.shape[1]
                col_sums = just_onehot.sum(axis=0)
                largest_indices = np.argsort(-1*col_sums)[:n]
                
                for id,val in enumerate(largest_indices):
                      re_ordered_jhot[:,id] = just_onehot[:,val]
                
                final_features = final[:,0].reshape([-1, 1])
                
                # storing the original ordering for invoking inverse transform
                self.ordering.append(largest_indices)
                
                values += [final_features, re_ordered_jhot]
                
                mixed_counter = mixed_counter + 1
    
            else:
                # for categorical columns, standard one-hot-encoding is applied where categories are in descending order of frequency by default
                self.ordering.append(None)
                col_t = np.zeros([len(data), info['size']])
                idx = list(map(info['i2s'].index, current))
                col_t[np.arange(len(data)), idx] = 1
                values.append(col_t)
                
        return np.concatenate(values, axis=1)

    def inverse_transform(self, data):
        
        # stores the final inverse transformed generated data 
        data_t = np.zeros([len(data), len(self.meta)])
        
        # used to iterate through the columns of the raw generated data
        st = 0

        # iterating through original column information
        for id_, info in enumerate(self.meta):
            if info['type'] == "continuous":
                
                # obtaining the generated normalized values and clipping for stability
                u = data[:, st]
                u = np.clip(u, -1, 1)
                
                # obtaining the one-hot-encoding of the modes representing the normalized values
                v = data[:, st + 1:st + 1 + np.sum(self.components[id_])]
                
                # re-ordering the modes as per their original ordering
                order = self.ordering[id_] 
                v_re_ordered = np.zeros_like(v)
                for id,val in enumerate(order):
                    v_re_ordered[:,val] = v[:,id]
                v = v_re_ordered

                # ensuring un-used modes are represented with -100 such that they can be ignored when computing argmax
                v_t = np.ones((data.shape[0], self.n_clusters)) * -100
                v_t[:, self.components[id_]] = v
                v = v_t
                
                # obtaining approriate means and stds as per the appropriately selected mode for each data point based on fitted bgm model
                means = self.model[id_].means_.reshape([-1])
                stds = np.sqrt(self.model[id_].covariances_).reshape([-1])
                p_argmax = np.argmax(v, axis=1)
                std_t = stds[p_argmax]
                mean_t = means[p_argmax]
                
                # executing the inverse transformation 
                tmp = u * 4 * std_t + mean_t
                
                data_t[:, id_] = tmp
                
                # moving to the next set of columns in the raw generated data in correspondance to original column information
                st += 1 + np.sum(self.components[id_])
                
            elif info['type'] == "mixed":
                
                # obtaining the generated normalized values and corresponding modes
                u = data[:, st]
                u = np.clip(u, -1, 1)
                full_v = data[:,(st+1):(st+1)+len(info['modal'])+np.sum(self.components[id_])]
                
                # re-ordering the modes as per their original ordering
                order = self.ordering[id_]
                full_v_re_ordered = np.zeros_like(full_v)
                for id,val in enumerate(order):
                    full_v_re_ordered[:,val] = full_v[:,id]
                full_v = full_v_re_ordered                
                
                # modes of categorical component
                mixed_v = full_v[:,:len(info['modal'])]
                
                # modes of continuous component
                v = full_v[:,-np.sum(self.components[id_]):]

                # similarly ensuring un-used modes are represented with -100 to be ignored while computing argmax
                v_t = np.ones((data.shape[0], self.n_clusters)) * -100
                v_t[:, self.components[id_]] = v
                v = np.concatenate([mixed_v,v_t], axis=1)       
                p_argmax = np.argmax(v, axis=1)

                # obtaining the means and stds of the continuous component using second fitted bgm model
                means = self.model[id_][1].means_.reshape([-1]) 
                stds = np.sqrt(self.model[id_][1].covariances_).reshape([-1]) 

                # used to store the inverse-transformed data points
                result = np.zeros_like(u)

                for idx in range(len(data)):
                    # in case of categorical mode being selected, the mode value itself is simply assigned 
                    if p_argmax[idx] < len(info['modal']):
                        argmax_value = p_argmax[idx]
                        result[idx] = float(list(map(info['modal'].__getitem__, [argmax_value]))[0])
                    else:
                        # in case of continuous mode being selected, similar inverse-transform for purely numeric values is applied
                        std_t = stds[(p_argmax[idx]-len(info['modal']))]
                        mean_t = means[(p_argmax[idx]-len(info['modal']))]
                        result[idx] = u[idx] * 4 * std_t + mean_t
            
                data_t[:, id_] = result

                st += 1 + np.sum(self.components[id_]) + len(info['modal'])
                
            else:
                # reversing one hot encoding back to label encoding for categorical columns 
                current = data[:, st:st + info['size']]
                idx = np.argmax(current, axis=1)
                data_t[:, id_] = list(map(info['i2s'].__getitem__, idx))
                st += info['size']
        return data_t

class ImageTransformer():

    """
    Transformer responsible for translating data rows to images and vice versa

    Variables:
    1) side -> height/width of the image

    Methods:
    1) __init__() -> initializes image transformer object with given input
    2) transform() -> converts tabular data records into square image format
    3) inverse_transform() -> converts square images into tabular format

    """
    
    def __init__(self, side):
    
        self.height = side
            
    def transform(self, data):
        
        if self.height * self.height > len(data[0]):
            # tabular data records are padded with 0 to conform to square shaped images
            padding = torch.zeros((len(data), self.height * self.height - len(data[0]))).to(data.device)
            data = torch.cat([data, padding], axis=1)

        return data.view(-1, 1, self.height, self.height)

    def inverse_transform(self, data):
        
        data = data.view(-1, self.height * self.height)

        return data


