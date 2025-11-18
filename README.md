# hermanha-ml

```git clone https://github.com/INDA25PlusPlus/mnist.git```


run:
```
python -m venv .venv
pip install -r requirements.txt
```

To see the accuracy of the models you can scroll down to evaluate accuracy on saved model, in the train file and run it. It will load the saved weights and evaluate the model on test data. You wont get anythin different from me, so i dont know why you would like to do so. But go ahead!

Everything you need is in train.ipynb. There you have the different generations of the model i have created. From the simple model using just sigmoid and mse. doing a forward and back propagation for each image in the dataset. The simple model got a result of 91.8 % accuracy TJOHOOOO!!!

I then wanted to do some improvements and read that the softmax activation is better suited for classification tasks because it makes sure the sum of all the output probabilities is 1. Which is reasonable, you can't be 90% an image shows a 1 and 30% sure it shows a 2???? For softmax, its also common to use cross entropy loss as it values how confident the model is in its answer. If its only 80% certain it's right, thats not very good!

So the model using one dense layer with sigmoid and one output layer using softmax, together with cross entropy loss resulted in a higher accuracy of 97.9% TJOHOOOOO!!!!

This is starting to go well, so why not add more groundbraking, researched, innovative improvements??? Lets add learning rate decay. As you get closer to a local minimum, its common to start making smaller steps in the gradient descent, reasonable as you dont want to jump over the local minima! I added a small decay, decreasing the learning rate with 5% each epoch. This resulted in an EVEN.......... worse.... model??? well, i guess randomness has it's part in this aswell. But the result with the exact same layout as the softmax + cross entropy model but with added learning rate decay, resulted in an accuracy of 97.7%

I couldn't end with the last improvement making the model, slightly, just slightly worse, sooo i tried playing around with layers and model structure. And tried adding a third layer, and using the common beautiful numbers of 256, 128 and 10 for their sizes. First layer 256, second layer 128 and third output layer 10. (why theese, i dont know. I've always wondered myself why they have to be so good looking, probably doesnt matter). This more complex model seemed to have improved our score a bit with an accuracy of exactly 98%. And you said it yourselves, thats a "väldigt bra resultat". Thank you very much! that's it for me. Would be intersting to make a CNN though, might come back!

## FINAL RESULT: 98% Accuracy

<small>Sorry for not making an inference, that would be nice:D</small>
