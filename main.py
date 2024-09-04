import train
import eval

if __name__ == '__main__':
    train.train_model(num_epochs=10)
    eval.evaluate_model()
