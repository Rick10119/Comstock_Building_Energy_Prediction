import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.discriminant_analysis import StandardScaler

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU available")

upgrade_selected_data_df = pd.read_parquet('C:/Users/3.22/VS Code 2024/Demand-Response-Potential-Assessment-through-AI-and-Simulation-main/downloaded_data/selected_data_new_upgrade1_seasons.parquet')
#Clean unused features
unused_features = ['bldg_id','in.airtightness..m3_per_m2_s','applicability','in.building_subtype','in.wall_construction_type','in.window_to_wall_ratio_category',
                    'in.window_type',]
upgrade_selected_data_df = upgrade_selected_data_df.drop(columns=unused_features)

#Clean unused features
str_to_num_col = ['in.aspect_ratio','in.number_of_stories','in.weekday_opening_time..hr',
                  'in.weekday_operating_hours..hr','in.weekend_opening_time..hr','in.weekend_operating_hours..hr',
                  'in.year_built']


# 提取特征和标签
selected_output_df = pd.read_excel('C:/Users/3.22/VS Code 2024/Demand-Response-Potential-Assessment-through-AI-and-Simulation-main/selected_output.xlsx')
X = upgrade_selected_data_df.drop(columns=selected_output_df['Output Name'])
X = X[upgrade_selected_data_df['season'] == 'summer']
X['Short-Radiation [W/m2]'] = X['Global Horizontal Radiation [W/m2]'] + X['Direct Normal Radiation [W/m2]'] + X['Diffuse Horizontal Radiation [W/m2]']
X = X.drop(columns = ['season','Global Horizontal Radiation [W/m2]','Direct Normal Radiation [W/m2]','Diffuse Horizontal Radiation [W/m2]'])
print(X.columns)
feature_names = X.columns

y = upgrade_selected_data_df[upgrade_selected_data_df['season'] == 'summer']
y = y[selected_output_df['Output Name']]
y = y['out.qoi.maximum_daily_use_summer..kw']


print(X.shape)
#Replace NAN values with data mean and make all values double
y = y.fillna(y.mean())
X[str_to_num_col] = X[str_to_num_col].astype('double')
X = X.select_dtypes(include=[np.number])
X = X.values.astype('double')
y = y.values.astype('double')

#Replace NAN values with data mean
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

print(X.shape)
print(y.shape)
#Remove bottom 5% outliers
X = X[y > np.percentile(y, 5)]
y = y[y > np.percentile(y, 5)]
X = X[:170000]
y = y[:170000]


# Normalized
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Features Selection
n_components = 5
selector = SelectKBest(score_func=f_regression, k=n_components)
selector.fit(X,y)
X = selector.transform(X)
# selector.fit(X_train,y_train)
# X_train = selector.transform(X_train)
# X_test = selector.transform(X_test)

#Printing features name
selected_indices = selector.get_support(indices=True)
selected_features =  [feature_names[i] for i in selected_indices]
print(f'Selected features: {selected_features}')
print(f'Selected indices: {selected_indices}') 


# Printing out each features F-values and p-values
feature_scores = selector.scores_
feature_pvalues = selector.pvalues_
for i, (score, pvalue) in enumerate(zip(feature_scores, feature_pvalues)):
    print(f'Feature {i}: F-value = {score:.3f}, p-value = {pvalue:.3f}')


# 5. Create sequences (e.g., 24-hour windows, 1 datapoint is 15 minutes)
def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length + 1):
        Xs.append(X[i:i+seq_length])
        ys.append(y[i:i+seq_length])
    return np.array(Xs), np.array(ys)


X_seq, y_seq = create_sequences(X, y, seq_length=96)
print(X_seq.shape[2])

# 6. Reshape for Conv2D: (samples, height, width, channels)
# Here, Total Samples = N, height=96 (timesteps), width=2 (features), channels=1
X_seq = X_seq.reshape((X_seq.shape[0],1, X_seq.shape[1], X_seq.shape[2]))

# 7. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
print(X_test.shape)


#Turning the data into tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)


# Create data loaders
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)



class HVACCNN(nn.Module):
    def __init__(self, input_shape, output_length):
        super(HVACCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(2, 2), padding='valid') #128
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(2, 1), padding='valid') #256
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=(2, 1), padding='valid') #256
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(2, 1), padding='valid') #256
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=(2, 1), padding='valid') #256
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=(2, 1), padding='valid') #256
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=(2, 1), padding='valid') #256
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=(2, 1), padding='valid') #256
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 128, kernel_size=(2, 1), padding='valid') #256
        self.bn9 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        
        # Calculate flattened dimension
        with torch.no_grad():
            self.flatten_dim = self._get_flatten_dim(input_shape)
            
        self.fc1 = nn.Linear(self.flatten_dim, 128)#256
        self.fc2 = nn.Linear(128, output_length)
        self.relu = nn.ReLU()

    def _get_flatten_dim(self, input_shape):
        dummy = torch.zeros(1, *input_shape)
        dummy = self.conv1(dummy)
        dummy = self.conv2(dummy)
        dummy = self.conv3(dummy)
        dummy = self.conv4(dummy)
        dummy = self.conv5(dummy)
        dummy = self.conv6(dummy)
        dummy = self.conv7(dummy)
        dummy = self.conv8(dummy)
        dummy = self.conv9(dummy)
        dummy = self.pool(dummy)
        return dummy.view(1, -1).shape[1]

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))
        x = self.relu(self.bn9(self.conv9(x)))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model
input_shape = (1, X_train.shape[2], X_train.shape[3])  # (channels, timesteps, features)
model = HVACCNN(input_shape, output_length=96).to(device)
#model = HVACCNN(X_train.shape[1]), output_length=1).to(device)
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4,weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# Training function
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs):
    model.train()
    best_loss = float('inf')
    patience = 5
    no_improve_epochs = 0
    
    # Create history dictionary to track metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        # Create progress bar for batches
        from tqdm import tqdm
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=True)
        
        for inputs, targets in train_loop:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Update running loss
            running_loss += loss.item() * inputs.size(0)
            
            # Update progress bar description
            avg_loss = running_loss / (train_loop.n * inputs.size(0)+ 1e-9)
            train_loop.set_postfix({
                'batch_loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Calculate epoch training loss
        epoch_train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = val_loss / len(test_loader.dataset)
        history['val_loss'].append(epoch_val_loss)
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{epochs} Summary:')
        print(f'Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 60)
        
        # Update scheduler
        scheduler.step(epoch_val_loss)
        
        # Early stopping and model saving
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Validation loss improved to {epoch_val_loss:.6f}. Saving model...')
        else:
            no_improve_epochs += 1
            print(f'Validation loss did not improve. Patience: {no_improve_epochs}/{patience}')
            if no_improve_epochs >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break
    
    
    return history

# Train the model
train_model(model, train_loader, test_loader, criterion, optimizer, epochs= 70)

# Load best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    return all_outputs, all_targets

# Evaluate
y_train_pred, y_train = evaluate_model(model, DataLoader(train_dataset, batch_size=batch_size))
y_test_pred, y_test = evaluate_model(model, DataLoader(test_dataset, batch_size=batch_size))


train_rmse = np.sqrt(mean_squared_error(y_train.flatten(), y_train_pred.flatten()))
test_rmse = np.sqrt(mean_squared_error(y_test.flatten(), y_test_pred.flatten()))

train_nrmse = train_rmse/np.mean(y_train.flatten())*100
test_nrmse = train_rmse/np.mean(y_test.flatten())*100

print(f'Train RMSE: {train_nrmse:.2f} %')
print(f'Test RMSE: {test_nrmse:.2f} %')

X_test = X_test.cpu().detach().numpy()

# 11. Plot example
import matplotlib.pyplot as plt
plt.plot(y_test[0,:50], label='Actual HVAC',linewidth=2)
plt.plot(y_test_pred[0,:50], label='Predicted HVAC',linewidth=2)
plt.xlabel('Number of Buildings')
plt.ylabel('Total Maximum Daily Use (kWh)')
plt.title('Maximum Daily Use in Highest Temperature Point in Summer')
plt.legend()
plt.grid(True)
plt.show()


# 获取排序后的索引
sorted_indices = np.argsort(y_test)
# 创建一个与 y_test 维度相同的数组，用于存储排名
y_test_order = np.empty_like(sorted_indices)
# 根据排序后的索引生成排名
y_test_order[sorted_indices] = np.arange(len(y_test))
# 将排名从 0-based 转换为 1-based
y_test_order = y_test_order + 1
# 对y_test_pred进行相同操作
sorted_indices = np.argsort(y_test_pred)
y_test_pred_order = np.empty_like(sorted_indices)
y_test_pred_order[sorted_indices] = np.arange(len(y_test_pred))
y_test_pred_order = y_test_pred_order + 1


# %% 画出预测值的每个点对应的排序，计算相关系数并画在图上
plt.figure(figsize=(10, 5))
plt.scatter(y_test_order, y_test_pred_order)
plt.xlabel('True Order')
plt.ylabel('Predicted Order')
plt.plot([0, len(y_test)], [0, len(y_test)], color='red')
plt.title(f'Correlation: {np.corrcoef(y_test_order, y_test_pred_order)[0, 1]:.2f}')
plt.rcParams.update({'font.size': 15})
plt.show()


#Weather data(Temperature)
plt.figure(figsize=(10, 5))
plt.plot(-np.cumsum(-y_test[sorted_indices]), label='True')
plt.plot(-np.cumsum(-y_test_pred[sorted_indices]), label='Predicted')
plt.xlabel('Number of Selected Buildings')
plt.ylabel('Cummulative Maximum Daily Use (kWh)')
plt.legend()
plt.rcParams.update({'font.size': 15})
plt.show()

