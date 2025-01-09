import os
import torch   
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler, autocast
from torchmetrics import Accuracy
from pathlib import Path

def train(model, dataloader, criterion, optimizer, num_epochs, checkpoint_dir):
    model.train()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    accuracy = Accuracy(task='binary').to(device)
    scaler = GradScaler()

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in tqdm(range(num_epochs)):
        training_loss = 0.0
        accuracy.reset()
        
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader)):
                labels = labels.float().to(device)
                inputs = inputs.to(device)

                """outputs = model(inputs)
                loss = criterion(outputs.view(-1), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_loss += loss.item()
                """

                # Forward pass with mixed-precision
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs.view(-1), labels)

                # Backward pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        
                # Update accuracy metric
                training_loss += loss.item()
                predicted = (outputs.view(-1) > 0.5).float()
                accuracy.update(predicted, labels)

                if batch_idx % 1000 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
                torch.cuda.empty_cache()  # Clear CUDA cache to prevent memory leaks

        epoch_loss = training_loss / len(dataloader)
        epoch_accuracy = accuracy.compute().item() * 100
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved at {checkpoint_path}')
    
    # Save final model
    MODEL_PATH = Path("checpoints")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = "final_checkpoint.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(),
                f=MODEL_SAVE_PATH)
    

    # final_checkpoint_path = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
    # torch.save(model.state_dict(), final_checkpoint_path)
    # print(f'Final checkpoint saved at {final_checkpoint_path}')

def main():
    pass

if __name__ == "__main__":
    main()