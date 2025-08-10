import torch
import copy
from datetime import datetime
import numpy as np
from sklearn.model_selection import GridSearchCV
import optuna
from typing import Dict, Any, Callable

class ContinualLearner:
    def __init__(self, model, performance_threshold=0.01, max_age_days=30):
        """
        Initialize the continual learner
        
        Args:
            model: Initial model
            performance_threshold: Performance improvement threshold
            max_age_days: Maximum model age in days
        """
        self.model = model
        self.performance_threshold = performance_threshold
        self.max_age_days = max_age_days
        self.best_model = copy.deepcopy(model)
        self.best_performance = 0.0
        self.last_update = datetime.now()
        self.update_count = 0
        
    def evaluate_and_update(self, val_loader, current_performance):
        """
        Evaluate model performance and decide whether to update
        
        Args:
            val_loader: Validation data loader
            current_performance: Current model performance
            
        Returns:
            bool: Whether the best model was updated
        """
        # If performance significantly improves, update the best model
        if current_performance > self.best_performance + self.performance_threshold:
            self.best_model = copy.deepcopy(self.model)
            self.best_performance = current_performance
            self.last_update = datetime.now()
            self.update_count += 1
            print(f"Model updated. New best performance: {current_performance:.4f}")
            return True
        return False
    
    def get_model_age(self):
        """
        Get the time since the model was last updated
        
        Returns:
            timedelta: Model age
        """
        return datetime.now() - self.last_update
    
    def should_retrain(self):
        """
        Decide whether retraining is needed based on model age
        
        Returns:
            bool: Whether retraining is needed
        """
        age = self.get_model_age()
        return age.days > self.max_age_days
    
    def get_best_model(self):
        """
        Get the best model
        
        Returns:
            Best model
        """
        return self.best_model
    
    def get_status(self):
        """
        Get the continual learner status
        
        Returns:
            dict: Status information
        """
        return {
            'best_performance': self.best_performance,
            'last_update': self.last_update,
            'model_age_days': self.get_model_age().days,
            'update_count': self.update_count,
            'should_retrain': self.should_retrain()
        }


class HyperparameterOptimizer:
    def __init__(self, model_class, param_grid=None):
        """
        Initialize the hyperparameter optimizer
        
        Args:
            model_class: Model class
            param_grid: Parameter grid (for grid search)
        """
        self.model_class = model_class
        self.param_grid = param_grid
        
    def grid_search(self, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1):
        """
        Grid search for hyperparameters
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv: Number of cross-validation folds
            scoring: Scoring standard
            n_jobs: Number of parallel jobs
            
        Returns:
            tuple: (Best parameters, Best score)
        """
        if self.param_grid is None:
            raise ValueError("Parameter grid must be provided for grid search")
            
        grid_search = GridSearchCV(
            estimator=self.model_class(),
            param_grid=self.param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs
        )
        grid_search.fit(X_train, y_train)
        return grid_search.best_params_, grid_search.best_score_
    
    def bayesian_optimization(self, objective_fn, n_trials=100, direction='maximize'):
        """
        Bayesian optimization using Optuna
        
        Args:
            objective_fn: Objective function
            n_trials: Number of trials
            direction: Optimization direction ('maximize' or 'minimize')
            
        Returns:
            tuple: (Best parameters, Best value)
        """
        def objective(trial):
            # Call the user-defined objective function
            return objective_fn(trial)
        
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params, study.best_value
    
    def adaptive_optimization(self, train_fn, evaluate_fn, initial_params=None, 
                             max_iterations=20, improvement_threshold=0.001):
        """
        Adaptive hyperparameter optimization
        
        Args:
            train_fn: Training function
            evaluate_fn: Evaluation function
            initial_params: Initial parameters
            max_iterations: Maximum number of iterations
            improvement_threshold: Improvement threshold
            
        Returns:
            dict: Optimization results
        """
        best_params = initial_params or {}
        best_score = -np.inf
        no_improvement_count = 0
        history = []
        
        for i in range(max_iterations):
            # Train model with current best parameters
            model = train_fn(best_params)
            
            # Evaluate model
            score = evaluate_fn(model)
            history.append((i, copy.deepcopy(best_params), score))
            
            print(f"Iteration {i+1}/{max_iterations}: Score = {score:.6f}")
            
            # Check if there is improvement
            if score > best_score + improvement_threshold:
                best_score = score
                no_improvement_count = 0
                print(f"New best score: {score:.6f}")
            else:
                no_improvement_count += 1
                
            # Early stopping if no improvement for several iterations
            if no_improvement_count >= 5:
                print("Early stopping due to no improvement")
                break
                
            # Adjust parameters (simplified implementation, more complex strategies needed in practice)
            best_params = self._adjust_parameters(best_params, i)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'history': history,
            'iterations': len(history)
        }
    
    def _adjust_parameters(self, params, iteration):
        """
        Adjust parameters (simplified implementation)
        """
        # This is an example implementation, in practice adjustments need to be made based on specific parameter types
        new_params = params.copy()
        for key, value in new_params.items():
            if isinstance(value, (int, float)) and key != 'batch_size':
                # Make small adjustments to numerical parameters
                adjustment = np.random.normal(0, 0.1 * abs(value))
                new_params[key] = max(0, value + adjustment)  # Ensure non-negative
                
        # Special handling for batch_size
        if 'batch_size' in new_params:
            batch_options = [16, 32, 64, 128]
            current_idx = batch_options.index(new_params['batch_size']) if new_params['batch_size'] in batch_options else 1
            new_idx = max(0, min(len(batch_options)-1, current_idx + np.random.choice([-1, 0, 1])))
            new_params['batch_size'] = batch_options[new_idx]
            
        return new_params


class ModelSelector:
    def __init__(self, models_dict: Dict[str, Any]):
        """
        Initialize the model selector
        
        Args:
            models_dict: Model dictionary {name: model_class}
        """
        self.models_dict = models_dict
        self.performance_history = {name: [] for name in models_dict.keys()}
        
    def evaluate_models(self, X_train, y_train, X_val, y_val, 
                       evaluation_metric: Callable = None):
        """
        Evaluate all models
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            evaluation_metric: Evaluation metric function
            
        Returns:
            dict: Model performance dictionary
        """
        results = {}
        
        for name, model_class in self.models_dict.items():
            try:
                # Train model
                model = model_class()
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train)
                elif hasattr(model, 'train'):
                    # For PyTorch models
                    self._train_pytorch_model(model, X_train, y_train)
                
                # Evaluate model
                if evaluation_metric:
                    score = evaluation_metric(model, X_val, y_val)
                else:
                    # Default to accuracy
                    if hasattr(model, 'predict'):
                        predictions = model.predict(X_val)
                        score = np.mean(predictions == y_val)
                    else:
                        score = 0
                
                results[name] = score
                self.performance_history[name].append(score)
                
                print(f"Model {name}: {score:.6f}")
                
            except Exception as e:
                print(f"Error evaluating model {name}: {str(e)}")
                results[name] = -np.inf
                
        return results
    
    def select_best_model(self, X_train, y_train, X_val, y_val, 
                         evaluation_metric: Callable = None):
        """
        Select the best model
        
        Returns:
            tuple: (Best model name, Best model instance, Performance score)
        """
        results = self.evaluate_models(X_train, y_train, X_val, y_val, evaluation_metric)
        
        # Select the best model
        best_model_name = max(results, key=results.get)
        best_score = results[best_model_name]
        
        # Instantiate the best model
        best_model = self.models_dict[best_model_name]()
        if hasattr(best_model, 'fit'):
            best_model.fit(X_train, y_train)
        elif hasattr(best_model, 'train'):
            self._train_pytorch_model(best_model, X_train, y_train)
            
        return best_model_name, best_model, best_score
    
    def _train_pytorch_model(self, model, X_train, y_train, epochs=10):
        """
        Train PyTorch model (simplified implementation)
        """
        model.train()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.BCELoss() if len(np.unique(y_train)) == 2 else torch.nn.CrossEntropyLoss()
        
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train) if len(np.unique(y_train)) == 2 else torch.LongTensor(y_train)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            loss.backward()
            optimizer.step()


class PerformanceMonitor:
    def __init__(self, model, metrics=None):
        """
        Initialize the performance monitor
        
        Args:
            model: Model to monitor
            metrics: List of evaluation metrics
        """
        self.model = model
        self.metrics = metrics or ['accuracy', 'precision', 'recall', 'f1']
        self.history = []
        self.last_evaluation = None
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Returns:
            dict: Performance metrics
        """
        from sklearn import metrics
        
        # Get prediction results
        if hasattr(self.model, 'predict'):
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        else:
            # Assume it's a PyTorch model
            self.model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test)
                y_pred_proba = torch.sigmoid(self.model(X_test_tensor)).squeeze().numpy()
                y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        results = {}
        
        if 'accuracy' in self.metrics:
            results['accuracy'] = metrics.accuracy_score(y_test, y_pred)
            
        if 'precision' in self.metrics:
            results['precision'] = metrics.precision_score(y_test, y_pred, zero_division=0)
            
        if 'recall' in self.metrics:
            results['recall'] = metrics.recall_score(y_test, y_pred, zero_division=0)
            
        if 'f1' in self.metrics:
            results['f1'] = metrics.f1_score(y_test, y_pred, zero_division=0)
            
        if 'auc' in self.metrics and y_pred_proba is not None:
            results['auc'] = metrics.roc_auc_score(y_test, y_pred_proba)
            
        # Record timestamp
        results['timestamp'] = datetime.now()
        
        # Save to history
        self.history.append(results)
        self.last_evaluation = results
        
        return results
    
    def get_performance_trend(self):
        """
        Get performance trend
        
        Returns:
            dict: Performance trend data
        """
        if len(self.history) < 2:
            return None
            
        trend = {}
        for metric in self.metrics:
            values = [record[metric] for record in self.history if metric in record]
            if len(values) >= 2:
                # Calculate trend (simple linear regression slope)
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                trend[metric] = slope
                
        return trend
    
    def plot_performance_history(self, metrics=None):
        """
        Plot performance history chart
        """
        import matplotlib.pyplot as plt
        
        if not self.history:
            print("No evaluation history available")
            return
            
        metrics = metrics or self.metrics
        timestamps = [record['timestamp'] for record in self.history]
        
        plt.figure(figsize=(12, 8))
        
        for i, metric in enumerate(metrics):
            if metric in self.history[0]:
                values = [record[metric] for record in self.history]
                plt.subplot(2, 2, i+1)
                plt.plot(timestamps, values, marker='o')
                plt.title(f'{metric.capitalize()} Over Time')
                plt.xlabel('Time')
                plt.ylabel(metric.capitalize())
                plt.xticks(rotation=45)
                plt.grid(True)
        
        plt.tight_layout()
        plt.show()
