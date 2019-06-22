class Boosting:
    def __init__(self,dataset,T,test_dataset):
        self.dataset = dataset
        self.T = T
        self.test_dataset = test_dataset
        self.alphas = None
        self.models = None
        self.accuracy = []
        self.predictions = None

    def fit(self):
        # Set the descriptive features and the target feature
        X = self.dataset.drop(["target"],axis=1)
        Y = self.dataset["target"].where(self.dataset["target"]==1,-1)
        # Initialize the weights of each sample with wi = 1/N and create a dataframe in which the evaluation is computed
        Evaluation = pd.DataFrame(Y.copy())
        Evaluation["weights"] = 1/len(self.dataset) # Set the initial weights w = 1/N
        
        # Run the boosting algorithm by creating T "weighted models"
        alphas = [] 
        models = []
                for t in range(self.T):
            # Train the Decision Stump(s)
            Tree_model = DecisionTreeClassifier(criterion="entropy",max_depth=1) # Mind the deth one --> Decision Stump
            
            # We know that we must train our decision stumps on weighted datasets where the weights depend on the results of
            # the previous decision stumps. To accomplish that, we use the "weights" column of the above created 
            # "evaluation dataframe" together with the sample_weight parameter of the fit method.
            # The documentation for the sample_weights parameter sais: "[...] If None, then samples are equally weighted."
            # Consequently, if NOT None, then the samples are NOT equally weighted and therewith we create a WEIGHTED dataset 
            # which is exactly what we want to have.
            model = Tree_model.fit(X,Y,sample_weight=np.array(Evaluation["weights"])) 
            
            # Append the single weak classifiers to a list which is later on used to make the 
            # weighted decision
            models.append(model)
            predictions = model.predict(X)
            score = model.score(X,Y)
            # Add values to the Evaluation DataFrame
            Evaluation["predictions"] = predictions
            Evaluation["evaluation"] = np.where(Evaluation["predictions"] == Evaluation["target"],1,0)
            Evaluation["misclassified"] = np.where(Evaluation["predictions"] != Evaluation["target"],1,0)
            # Calculate the misclassification rate and accuracy
            accuracy = sum(Evaluation["evaluation"])/len(Evaluation["evaluation"])
            misclassification = sum(Evaluation["misclassified"])/len(Evaluation["misclassified"])
            # Caclulate the error
            err = np.sum(Evaluation["weights"]*Evaluation["misclassified"])/np.sum(Evaluation["weights"]) 
            # Calculate the alpha values
            alpha = np.log((1-err)/err)
            alphas.append(alpha)
            # Update the weights wi --> These updated weights are used in the sample_weight parameter
            # for the training of the next decision stump. 
            Evaluation["weights"] *= np.exp(alpha*Evaluation["misclassified"])
            #print("The Accuracy of the {0}. model is : ".format(t+1),accuracy*100,"%")
            #print("The missclassification rate is: ",misclassification*100,"%")