Machine Learning Chatbot Project
This project involves building a simple chatbot using TensorFlow and Keras. The chatbot can respond to specific questions about the user.

Project Structure
main.py: Main script to run the chatbot.
README.md: Project documentation.
Requirements
Python 3.x
TensorFlow
Pandas
Numpy
You can install the required libraries using:

bash
نسخ الكود
pip install tensorflow pandas numpy
Dataset
The dataset consists of a set of questions and answers about the user. The questions are used to train the model to recognize and respond correctly.

python
نسخ الكود
dataset = {
  "What is your name?": "my name is abdullah",
  "how old are you? ": "21",
  "what are you doing?": "I am student",
  "what do you study?": "I am studying artificial intelligence",
  "What university do you study at?": "Banha university",
  "What is the expected graduation year?": "2024",
  "What skills do you have?" : "C++ and Python programming languages, Ability to solve programming problems, Ability to analyze data and apply artificial intelligence techniques on it Using data analysis libraries such as Pandas and Numpy and Matplotlib data analysis using Python, Excel, and Power BI Machine Learning techniques such as linear regression, logistic regression, decision trees and KNN Good knowledge of neural networks, deep learning and NLP",
  "What do you do in your free time?": "I play football, watch football matches, read books, and learn new skills"
}
Model Architecture
The model uses a simple neural network architecture:

Embedding Layer
Global Average Pooling Layer
Dense Layer with ReLU activation
Output Dense Layer with Softmax activation
python
نسخ الكود
model = tf.keras.Sequential([
  tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=32, input_length=max_length),
  tf.keras.layers.GlobalAveragePooling1D(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(len(answers), activation='softmax')
])
Training the Model
The model is trained using sparse_categorical_crossentropy loss and adam optimizer for 30 epochs.

python
نسخ الكود
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded_questions, np.array(range(len(questions))), epochs=30)
Running the Chatbot
You can interact with the chatbot by running the chatbot function. It will prompt you to ask questions, and it will respond based on the trained model.

python
نسخ الكود
def chatbot():
    while True:
        user_input = input("ask me ")
        if user_input == 'thanks':
            print('your welcome')
            break
        else:
            user_sequence = tokenizer.texts_to_sequences([user_input])
            padded_user_sequence = tf.keras.preprocessing.sequence.pad_sequences(user_sequence, maxlen=max_length, padding='post')
            prediction = np.argmax(model.predict(padded_user_sequence))
            print(answers[prediction])

# Run the chatbot
chatbot()
Example Interaction
vbnet
نسخ الكود
ask me what do you study
1/1 [==============================] - 0s 81ms/step
I am studying artificial intelligence
ask me What is your name
1/1 [==============================] - 0s 17ms/step
my name is abdullah
ask me thanks
your welcome
License
This project is licensed under the MIT License.

This README file provides a clear overview of the project, instructions for setting up the environment, and examples of how to run and interact with the chatbot.






