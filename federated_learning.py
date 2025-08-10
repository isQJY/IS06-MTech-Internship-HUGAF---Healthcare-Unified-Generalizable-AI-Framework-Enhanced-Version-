import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
from typing import List, Dict, Any

class FederatedClient:
    def __init__(self, client_id: str, model: nn.Module, client_data: DataLoader, 
                 learning_rate: float = 0.001):
        """
        Initialize federated learning client
        
        Args:
            client_id: Client ID
            model: Client model
            client_data: Client local data
            learning_rate: Learning rate
        """
        self.client_id = client_id
        self.model = model
        self.client_data = client_data
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def train_local_model(self, epochs: int = 5, verbose: bool = False) -> float:
        """
        Train model on local data
        
        Args:
            epochs: Number of training epochs
            verbose: Whether to print training information
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(self.client_data):
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward propagation
                self.optimizer.zero_grad()
                output = self.model(data)
                
                # Calculate loss (assuming binary classification problem)
                criterion = nn.BCELoss()
                loss = criterion(output.squeeze(), target.float().squeeze())
                
                # Backpropagation
                loss.backward()
                self.optimizer.step()
                
                # Accumulate loss
                batch_size = data.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_samples += batch_size
                
                if verbose and batch_idx % 10 == 0:
                    print(f'Client {self.client_id} - Epoch {epoch+1}/{epochs} - Batch {batch_idx} - Loss: {loss.item():.6f}')
            
            total_loss += epoch_loss
            total_samples += epoch_samples
            
            if verbose:
                avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0
                print(f'Client {self.client_id} - Epoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.6f}')
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        return avg_loss
    
    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get model weights
        
        Returns:
            Model weights dictionary
        """
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def update_model_weights(self, global_weights: Dict[str, torch.Tensor]):
        """
        Update model weights
        
        Args:
            global_weights: Global model weights
        """
        self.model.load_state_dict(global_weights)


class FederatedServer:
    def __init__(self, global_model: nn.Module):
        """
        Initialize federated learning server
        
        Args:
            global_model: Global model
        """
        self.global_model = global_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model.to(self.device)
        
    def aggregate_weights(self, client_weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client weights (using FedAvg algorithm)
        
        Args:
            client_weights: List of client weights
            
        Returns:
            Aggregated weights
        """
        if not client_weights:
            raise ValueError("Client weights list is empty")
            
        # Initialize aggregated weights
        aggregated_weights = {}
        num_clients = len(client_weights)
        
        # Aggregate parameters for each name
        for name in client_weights[0].keys():
            # Stack all clients' parameters with the same name
            stacked_weights = torch.stack([client_weights[i][name] for i in range(num_clients)])
            # Calculate average
            aggregated_weights[name] = torch.mean(stacked_weights, dim=0)
            
        return aggregated_weights
    
    def update_global_model(self, aggregated_weights: Dict[str, torch.Tensor]):
        """
        Update global model
        
        Args:
            aggregated_weights: Aggregated weights
        """
        self.global_model.load_state_dict(aggregated_weights)
    
    def distribute_global_weights(self) -> Dict[str, torch.Tensor]:
        """
        Distribute global model weights to clients
        
        Returns:
            Global model weights
        """
        return {name: param.data.clone() for name, param in self.global_model.named_parameters()}


class FederatedLearningSystem:
    def __init__(self, global_model: nn.Module, clients: List[FederatedClient], 
                 server: FederatedServer = None):
        """
        Initialize federated learning system
        
        Args:
            global_model: Global model
            clients: List of clients
            server: Server (optional, if not provided it will be created automatically)
        """
        self.global_model = global_model
        self.clients = clients
        
        if server is None:
            self.server = FederatedServer(global_model)
        else:
            self.server = server
    
    def federated_training_round(self, local_epochs: int = 5, verbose: bool = False) -> Dict[str, Any]:
        """
        Execute one round of federated training
        
        Args:
            local_epochs: Number of local training epochs for clients
            verbose: Whether to print detailed information
            
        Returns:
            Training results dictionary
        """
        if verbose:
            print("Starting federated training round...")
        
        # 1. Distribute global model weights to all clients
        global_weights = self.server.distribute_global_weights()
        for client in self.clients:
            client.update_model_weights(global_weights)
        
        if verbose:
            print(f"Distributed global weights to {len(self.clients)} clients")
        
        # 2. Client local training
        client_weights = []
        client_losses = []
        
        for i, client in enumerate(self.clients):
            if verbose:
                print(f"Training client {i+1}/{len(self.clients)} ({client.client_id})...")
            
            # Local training
            loss = client.train_local_model(epochs=local_epochs, verbose=verbose)
            client_losses.append(loss)
            
            # Collect client weights
            weights = client.get_model_weights()
            client_weights.append(weights)
        
        # 3. Aggregate client weights
        aggregated_weights = self.server.aggregate_weights(client_weights)
        
        # 4. Update global model
        self.server.update_global_model(aggregated_weights)
        
        # Calculate statistics
        avg_loss = np.mean(client_losses) if client_losses else 0
        std_loss = np.std(client_losses) if len(client_losses) > 1 else 0
        
        result = {
            'avg_loss': avg_loss,
            'std_loss': std_loss,
            'client_losses': client_losses,
            'aggregated_weights': aggregated_weights
        }
        
        if verbose:
            print(f"Federated training round completed. Average loss: {avg_loss:.6f} ± {std_loss:.6f}")
        
        return result
    
    def run_federated_training(self, num_rounds: int = 10, local_epochs: int = 5, 
                               verbose: bool = False) -> List[Dict[str, Any]]:
        """
        Run multiple rounds of federated training
        
        Args:
            num_rounds: Number of federated training rounds
            local_epochs: Number of local training epochs per round
            verbose: Whether to print detailed information
            
        Returns:
            List of training results for each round
        """
        results = []
        
        for round_idx in range(num_rounds):
            if verbose:
                print(f"\n=== Federated Training Round {round_idx + 1}/{num_rounds} ===")
            
            # Execute one training round
            round_result = self.federated_training_round(
                local_epochs=local_epochs, 
                verbose=verbose
            )
            round_result['round'] = round_idx + 1
            results.append(round_result)
            
            if verbose:
                print(f"Round {round_idx + 1} completed. Avg loss: {round_result['avg_loss']:.6f}")
        
        return results


# 差分隐私机制
class DifferentiallyPrivateClient(FederatedClient):
    def __init__(self, client_id: str, model: nn.Module, client_data: DataLoader,
                 learning_rate: float = 0.001, noise_multiplier: float = 1.0, 
                 max_grad_norm: float = 1.0):
        """
        Initialize differentially private client
        
        Args:
            client_id: Client ID
            model: Client model
            client_data: Client local data
            learning_rate: Learning rate
            noise_multiplier: Noise multiplier (controls privacy budget)
            max_grad_norm: Maximum gradient norm (for gradient clipping)
        """
        super().__init__(client_id, model, client_data, learning_rate)
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
    
    def train_local_model(self, epochs: int = 5, verbose: bool = False) -> float:
        """
        Train model on local data (with differential privacy noise)
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(self.client_data):
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward propagation
                self.optimizer.zero_grad()
                output = self.model(data)
                
                # Calculate loss
                criterion = nn.BCELoss()
                loss = criterion(output.squeeze(), target.float().squeeze())
                
                # Backpropagation
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Add noise to gradients (differential privacy)
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is not None:
                            noise = torch.normal(
                                mean=0, 
                                std=self.noise_multiplier * self.max_grad_norm, 
                                size=param.grad.shape
                            ).to(self.device)
                            param.grad += noise / len(self.client_data)
                
                # Update parameters
                self.optimizer.step()
                
                # Accumulate loss
                batch_size = data.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_samples += batch_size
                
                if verbose and batch_idx % 10 == 0:
                    print(f'DP Client {self.client_id} - Epoch {epoch+1}/{epochs} - Batch {batch_idx} - Loss: {loss.item():.6f}')
            
            total_loss += epoch_loss
            total_samples += epoch_samples
            
            if verbose:
                avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0
                print(f'DP Client {self.client_id} - Epoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.6f}')
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        return avg_loss


# 联邦学习评估器
class FederatedEvaluator:
    def __init__(self, model: nn.Module):
        """
        Initialize federated evaluator
        
        Args:
            model: Model to evaluate
        """
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Performance metrics dictionary
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # Calculate loss
                criterion = nn.BCELoss()
                loss = criterion(output.squeeze(), target.float().squeeze())
                total_loss += loss.item() * data.size(0)
                
                # Calculate accuracy
                pred = (output.squeeze() > 0.5).float()
                correct += pred.eq(target.float()).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / total if total > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
