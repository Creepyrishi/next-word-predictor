#### 1
# Next Word Predictor (LSTM)

PyTorch LSTM model for next-word prediction using pickup lines. Currently underperforms due to training and data issues.

## Training Results
Trained 49 epochs (Adam, lr=0.001). Recent losses:
- Epoch 45: 0.3571
- Epoch 46: 0.4156
- Epoch 47: 0.5037
- Epoch 48: 0.5370
- Epoch 49: 0.2698

Loss fluctuates; predictions often incorrect.

## Example
**Input**: `why you pay`  
**Output**: `why you pay know pay seen for are pay have to a from you me`  
Output is repetitive and incoherent.

## Challenges
- Unstable model performance.
- Text preprocessing issues (emojis, rare words).
- Limited dataset quality.

## Future Improvements
- Cleaner/larger dataset.
- Better preprocessing, validation.

## Mistakes I made while making this model
- Softmaxing 
- Not doing this ```torch.argmax(y, dim=1)```
I have not yet gone deep why can we work with one hot encodded problem directly

#### 2
Not yet completed
