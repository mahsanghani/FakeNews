import pickle

var = input("Text: ")
print("Confirm: " + str(var))


# prediction function
def detecting_fake_news(var):
# best model
    load_model = pickle.load(open('final_model.sav', 'rb'))
    prediction = load_model.predict([var])
    prob = load_model.predict_proba([var])

    return (print("statement: ",prediction[0]),
        print("probability: ",prob[0][1]))

if __name__ == '__main__':
    detecting_fake_news(var)