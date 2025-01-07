import os
import torch   

def train(model, dataloader, criterion, optimizer, num_epochs, checkpoint_dir):
    model.train()

    for epoch in range(num_epochs):
        training_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(dataloader):

            labels = labels.float().to('cuda' if torch.cuda.is_available() else 'cpu')
            inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {training_loss/len(dataloader):.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved at {checkpoint_path}')

def main():
    pass

if __name__ == "__main__":
    main()