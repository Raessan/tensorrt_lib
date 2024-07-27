import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        # Convolutional layers for input (3, 100, 100)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers for the second input (2, 50)
        self.fc1_input2 = nn.Linear(2 * 50, 12)

        # Fully connected layers for the third input (25)
        self.fc1_input3 = nn.Linear(25, 12)

        # Calculate the output size after the conv layers
        final_feature_map_size = 100 // (2**2)  # After 2 pooling layers
        
        # Fully connected layers after concatenating all features
        fc_input_dim = (16 * final_feature_map_size * final_feature_map_size) + 12 + 12
        self.fc1_out1 = nn.Linear(fc_input_dim, 12)
        self.fc2_out1 = nn.Linear(12, 10)  # Output 1

        self.fc1_out2 = nn.Linear(fc_input_dim, 12)
        self.fc2_out2 = nn.Linear(12, 5)  # Output 2

    def forward(self, x1, x2, x3):
        # Process image input (3, 100, 100) through conv layers
        x1 = F.relu(self.conv1(x1))
        x1 = self.pool(x1)
        x1 = F.relu(self.conv2(x1))
        x1 = self.pool(x1)
        
        # Flatten and process through fully connected layers
        x1 = x1.view(x1.size(0), -1)  # Flatten for FC layer
        
        # Process second input (2, 50) through FC layers
        x2 = x2.view(x2.size(0), -1)  # Flatten for FC layer
        x2 = F.relu(self.fc1_input2(x2))
        
        # Process third input (25) through FC layers
        x3 = F.relu(self.fc1_input3(x3))
        
        # Concatenate all features
        x = torch.cat((x1, x2, x3), dim=1)
        
        # Output head 1
        out1 = F.relu(self.fc1_out1(x))
        out1 = self.fc2_out1(out1)
        
        # Output head 2
        out2 = F.relu(self.fc1_out2(x))
        out2 = self.fc2_out2(out2)
        
        return out1, out2

    
if __name__ == "__main__":
    model = DummyModel()
    model.eval()
    torch.save(model.state_dict(), "../../models/multi_io/weights.pt")
