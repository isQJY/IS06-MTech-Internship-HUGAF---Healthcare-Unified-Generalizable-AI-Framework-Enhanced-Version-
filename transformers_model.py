import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np

class MultimodalTransformer(nn.Module):
    def __init__(self, modality_dims, hidden_dim=768, num_heads=8, num_layers=6, num_classes=1):
        super(MultimodalTransformer, self).__init__()
        
        # Create projection layers for each modality
        self.modality_projections = nn.ModuleDict({
            modality: nn.Linear(dim, hidden_dim) 
            for modality, dim in modality_dims.items()
        })
        
        # Positional encoding
        self.position_embeddings = nn.Embedding(100, hidden_dim)  # Support up to 100 modalities
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
        
    def forward(self, modality_inputs):
        # Project each modality to unified dimension
        projected_inputs = []
        modality_names = []
        
        for modality, data in modality_inputs.items():
            if modality in self.modality_projections and data is not None:
                # Ensure input is at least 2D (batch_size, features)
                if len(data.shape) == 1:
                    data = data.unsqueeze(0)  # Add batch dimension
                
                # Apply projection layer
                projected = self.modality_projections[modality](data)
                
                # Ensure output is (batch_size, hidden_dim)
                if len(projected.shape) > 2:
                    # If output dimension is greater than 2, take average
                    projected = projected.mean(dim=1)
                elif len(projected.shape) == 1:
                    # If 1D, expand to 2D
                    projected = projected.unsqueeze(0)
                    
                projected_inputs.append(projected)
                modality_names.append(modality)
        
        # If no modality inputs, return default output
        if len(projected_inputs) == 0:
            batch_size = 1
            device = next(self.parameters()).device
            dummy_input = torch.zeros(batch_size, 1, self.modality_projections[list(self.modality_projections.keys())[0]].out_features).to(device)
            return torch.sigmoid(self.classifier(dummy_input.mean(dim=1)))
        
        # Concatenate all modalities
        if len(projected_inputs) > 1:
            combined = torch.cat(projected_inputs, dim=0)  # Concatenate along sequence dimension
        else:
            combined = projected_inputs[0]
        
        # Ensure combined is 3D tensor (batch_size, seq_length, hidden_dim)
        if len(combined.shape) == 2:
            # combined is now (batch_size*seq_length, hidden_dim)
            # Reshape to (batch_size, seq_length, hidden_dim)
            batch_size = 1  # Because we have one sample per modality
            seq_length = combined.shape[0]
            hidden_dim = combined.shape[1]
            combined = combined.view(batch_size, seq_length, hidden_dim)
        
        batch_size, seq_length, hidden_dim = combined.shape
        
        # Add positional encoding
        if seq_length <= 100:  # Ensure not exceeding maximum positional encoding length
            position_ids = torch.arange(seq_length, dtype=torch.long, device=combined.device)
            position_embeddings = self.position_embeddings(position_ids).unsqueeze(0)  # (1, seq_length, hidden_dim)
            # Expand position_embeddings to match combined's batch_size
            position_embeddings = position_embeddings.expand(batch_size, -1, -1)  # (batch_size, seq_length, hidden_dim)
        else:
            # If exceeding maximum length, only use first 100 positions
            position_ids = torch.arange(100, dtype=torch.long, device=combined.device)
            position_embeddings = self.position_embeddings(position_ids).unsqueeze(0)  # (1, 100, hidden_dim)
            position_embeddings = position_embeddings.expand(batch_size, -1, -1)  # (batch_size, 100, hidden_dim)
            # Truncate combined to 100
            combined = combined[:, :100, :]  # Truncate combined to 100
        
        # Apply Transformer encoder
        transformer_output = self.transformer_encoder(combined + position_embeddings)
        
        # Global pooling
        pooled_output = torch.mean(transformer_output, dim=1)
        
        # Classification
        output = self.classifier(pooled_output)
        
        return torch.sigmoid(output)


class MultimodalTransformerWithAttention(MultimodalTransformer):
    def __init__(self, modality_dims, hidden_dim=768, num_heads=8, num_layers=6, num_classes=1):
        super().__init__(modality_dims, hidden_dim, num_heads, num_layers, num_classes)
        
    def forward_with_attention(self, modality_inputs):
        # Project each modality to unified dimension
        projected_inputs = []
        modality_names = []
        
        for modality, data in modality_inputs.items():
            if modality in self.modality_projections and data is not None:
                # Ensure input is at least 2D (batch_size, features)
                if len(data.shape) == 1:
                    data = data.unsqueeze(0)  # Add batch dimension
                
                # Apply projection layer
                projected = self.modality_projections[modality](data)
                
                # Ensure output is (batch_size, hidden_dim)
                if len(projected.shape) > 2:
                    # If output dimension is greater than 2, take average
                    projected = projected.mean(dim=1)
                elif len(projected.shape) == 1:
                    # If 1D, expand to 2D
                    projected = projected.unsqueeze(0)
                    
                projected_inputs.append(projected)
                modality_names.append(modality)
        
        # If no modality inputs, return default output
        if len(projected_inputs) == 0:
            batch_size = 1
            device = next(self.parameters()).device
            dummy_input = torch.zeros(batch_size, 1, self.modality_projections[list(self.modality_projections.keys())[0]].out_features).to(device)
            return torch.sigmoid(self.classifier(dummy_input.mean(dim=1))), None, []
        
        # Concatenate all modalities
        if len(projected_inputs) > 1:
            combined = torch.cat(projected_inputs, dim=0)  # Concatenate along sequence dimension
        else:
            combined = projected_inputs[0]
        
        # Ensure combined is 3D tensor (batch_size, seq_length, hidden_dim)
        if len(combined.shape) == 2:
            # combined is now (batch_size*seq_length, hidden_dim)
            # Reshape to (batch_size, seq_length, hidden_dim)
            batch_size = 1  # Because we have one sample per modality
            seq_length = combined.shape[0]
            hidden_dim = combined.shape[1]
            combined = combined.view(batch_size, seq_length, hidden_dim)
        
        batch_size, seq_length, hidden_dim = combined.shape
        
        # Add positional encoding
        if seq_length <= 100:  # Ensure not exceeding maximum positional encoding length
            position_ids = torch.arange(seq_length, dtype=torch.long, device=combined.device)
            position_embeddings = self.position_embeddings(position_ids).unsqueeze(0)  # (1, seq_length, hidden_dim)
            # Expand position_embeddings to match combined's batch_size
            position_embeddings = position_embeddings.expand(batch_size, -1, -1)  # (batch_size, seq_length, hidden_dim)
        else:
            # If exceeding maximum length, only use first 100 positions
            position_ids = torch.arange(100, dtype=torch.long, device=combined.device)
            position_embeddings = self.position_embeddings(position_ids).unsqueeze(0)  # (1, 100, hidden_dim)
            position_embeddings = position_embeddings.expand(batch_size, -1, -1)  # (batch_size, 100, hidden_dim)
            # Truncate combined to 100
            combined = combined[:, :100, :]  # Truncate combined to 100
        
        # Record attention weights
        attention_weights = []
        
        # Apply Transformer encoder layers and collect attention weights
        x = combined + position_embeddings
        for layer in self.transformer_encoder.layers:
            # Get self-attention weights
            attn_output, attn_weights = layer.self_attn(x, x, x)
            attention_weights.append(attn_weights)
            # Apply remaining layer operations
            x = attn_output
            x = layer.dropout1(x)
            x = layer.norm1(x)
            # Feedforward network
            ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = x + ff_output
            x = layer.norm2(x)
        
        # Global pooling
        pooled_output = torch.mean(x, dim=1)
        
        # Classification
        output = self.classifier(pooled_output)
        
        return torch.sigmoid(output), attention_weights, modality_names


# Large language model text embedder
class LLMTextEmbedder:
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        """
        Initialize large language model text embedder
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # Set to evaluation mode
        self.model.eval()
        
    def get_embeddings(self, text):
        """
        Extract text embeddings using large language model
        """
        # Process input text
        if not text or len(text.strip()) == 0:
            # Return zero vector
            return np.zeros(768)
            
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=512
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the pooled output from the last layer
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.numpy().squeeze()
