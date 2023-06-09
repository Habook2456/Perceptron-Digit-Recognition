#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <GLFW/glfw3.h>

using namespace std;

class Perceptron
{
private:
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;
    double learningRate;

public:
    Perceptron(int numInputs, int numOutputs, double learningRate) : learningRate(learningRate)
    {
        weights.resize(numOutputs, std::vector<double>(numInputs));
        for (int i = 0; i < numOutputs; ++i)
        {
            for (int j = 0; j < numInputs; ++j)
            {
                weights[i][j] = 0;
            }
        }

        bias.resize(numOutputs, 0.0);
    }

    double activationFunction(double input)
    {
        if (input >= 0.0)
            return 1.0;
        else
            return 0.0;
    }

    std::vector<double> activate(const std::vector<double> &inputs)
    {
        std::vector<double> output(weights.size(), 0.0);

        for (int i = 0; i < weights.size(); ++i)
        {
            double sum = bias[i];
            for (int j = 0; j < weights[i].size(); ++j)
            {
                sum += inputs[j] * weights[i][j];
            }

            output[i] = activationFunction(sum);
        }

        return output;
    }

    void printOutputs(const std::vector<double> &outputs)
    {
        std::cout << "Outputs: ";
        for (const auto &output : outputs)
        {
            std::cout << output << " ";
        }
        std::cout << std::endl;
    }

    void train(const std::vector<std::vector<double>> &trainingSet, const std::vector<std::vector<double>> &labels, int numEpochs)
    {
        for (int epoch = 0; epoch < numEpochs; ++epoch)
        {
            std::cout << "Epoch " << epoch << std::endl;
            for (int i = 0; i < trainingSet.size(); ++i)
            {
                const std::vector<double> &inputs = trainingSet[i];
                const std::vector<double> &targetOutputs = labels[i];
                std::vector<double> outputs = activate(inputs);

                //printOutputs(outputs);
                // Calcular el error
                std::vector<double> error(outputs.size());
                for (int j = 0; j < outputs.size(); ++j)
                {
                    error[j] = targetOutputs[j] - outputs[j];
                }

                // Actualizar los pesos y el sesgo
                for (int j = 0; j < weights.size(); ++j)
                {
                    for (int k = 0; k < weights[j].size(); ++k)
                    {
                        weights[j][k] += learningRate * error[j] * inputs[k];
                    }
                    bias[j] += learningRate * error[j];
                }
            }
        }
    }
};

std::vector<std::vector<double>> readCSV(const std::string &filename, int numInputs)
{
    std::ifstream file(filename);
    std::vector<std::vector<double>> data;

    if (file)
    {
        std::string line;
        while (std::getline(file, line))
        {
            std::vector<double> row(numInputs);
            std::stringstream ss(line);
            std::string value;

            for (int i = 0; i < numInputs; ++i)
            {
                std::getline(ss, value, ',');
                row[i] = std::stod(value);
            }
            data.push_back(row);
        }

        file.close();
    }

    return data;
}

std::vector<std::vector<double>> createLabels(int numOutputs, int target)
{
    std::vector<std::vector<double>> labels(numOutputs, std::vector<double>(numOutputs, 0.0));
    for (int i = 0; i < numOutputs; ++i)
    {
        labels[i][target] = 1.0;
    }
    return labels;
}

void printLabels(const std::vector<std::vector<double>> &labels)
{
    for (int i = 0; i < labels.size(); ++i)
    {
        std::cout << "Label for training example " << i << ": ";
        for (int j = 0; j < labels[i].size(); ++j)
        {
            std::cout << labels[i][j] << " ";
        }
        std::cout << std::endl;
    }
}
void printTrainingSet(const std::vector<std::vector<double>> &trainingSet)
{
    for (int i = 0; i < trainingSet.size(); ++i)
    {
        std::cout << "Training example " << i << ": ";
        for (int j = 0; j < trainingSet[i].size(); ++j)
        {
            std::cout << trainingSet[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int main()
{
    int numInputs = 35;  // 7x5 píxeles
    int numOutputs = 10; // Dígitos del 0 al 9
    double learningRate = 0.5;
    int numEpochs = 1000;

    std::vector<std::vector<double>> trainingSet = readCSV("training_set.csv", numInputs);
    std::vector<std::vector<double>> labels;

    for (int i = 0; i < 10; ++i)
    {
        std::vector<std::vector<double>> digitLabels = createLabels(numOutputs, i);
        labels.push_back(digitLabels[0]);
    }

    Perceptron perceptron(numInputs, numOutputs, learningRate);
    perceptron.train(trainingSet, labels, numEpochs);
    printLabels(labels);
    std::vector<std::vector<double>> testSet = readCSV("test_set.csv", numInputs);

    for (const auto &inputs : testSet)
    {
        std::vector<double> outputs = perceptron.activate(inputs);
        double maxOutput = 0.0;
        int predictedDigit = -1;

        for (int i = 0; i < outputs.size(); ++i)
        {
            if (outputs[i] > maxOutput)
            {
                maxOutput = outputs[i];
                predictedDigit = i;
            }
        }

        std::cout << "Predicted Digit: " << predictedDigit << std::endl;
    }
    if (!glfwInit())
        return false;

    /* Create a windowed mode window and its OpenGL context */
    int wWidth = 500;
    int wHeight = 700;
    int sX = wWidth / 5;
    int sY = wHeight / 7;
    auto window = glfwCreateWindow(wWidth, wHeight, "TSP", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return false;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    /* Extra configurations */
    glClearColor(0, 0, 0, 1);
    //glfwSetWindowUserPointer(window, this);
    //glfwSetKeyCallback(window, keyCallback);
    //glfwSetMouseButtonCallback(window, mouseCallback);

    int testN = 0;
    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT);

        for (int k = 0; k < testSet[testN].size(); k++)
        {
            int i = k % 5;
            int j = k / 5;
                if (testSet[testN][k])
                {
                    glBegin(GL_QUADS);
                    glColor3f(1, 1, 1);
                    int tlx = i * sX;
                    int tly = j * sY;
                    auto openGlx = [wWidth](float x) {return 2.0f * x / (float)wWidth - 1; };
                    auto openGly = [wHeight](float y) {return -(2.0f * y / (float)wHeight - 1); };
                    glVertex2f(openGlx(tlx), openGly(tly));
                    glVertex2f(openGlx(tlx + sX), openGly(tly));
                    glVertex2f(openGlx(tlx + sX), openGly(tly + sY));
                    glVertex2f(openGlx(tlx), openGly(tly + sY));
                    glEnd();
                }
        }
        glfwSwapBuffers(window);
        glfwPollEvents();
        std::cin >> testN;
        testN = std::min(testN, (int)testSet.size() - 1);
        testN = std::max(testN, 0);
    }
    glfwTerminate();

    return 0;
}
