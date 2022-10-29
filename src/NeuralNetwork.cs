public class NeuralNetwork
{
  // I = input nodes, J = hidden layer nodes, K = output nodes
  public static int I = 4, J = 2, K = 2;
  private double bias = -1;
  // v_weights = matrix containing the weights between the input and hidden layer nodes.
  // w_weights = matrix containing the weights between the hidden and output layer nodes.
  private double[,] v_weights, w_weights;
  public NeuralNetwork()
  {
    this.v_weights = new double[J, I + 1];
    this.w_weights = new double[K, J + 1];
  }

  // NN expects the weight matrices to be passed into the constructor.
  public NeuralNetwork(double[,] v_weights, double[,] w_weights)
  {
    this.v_weights = v_weights;
    this.w_weights = w_weights;
  }

  // Give the NN an input pattern/vector
  // pattern = [error_x, error_y, angle_x, angle_y]
  // and method will return the output vector
  // output = [motor_1_output, motor_2_output]
  public double[] predict(double[] pattern)
  {
    double[] predicted = new double[K];
    for (int k = 0; k < K; k++)
    {
      predicted[k] = outputForKthUnitValue(k, pattern);
    }
    
    return predicted;
  }

  private double outputForKthUnitValue(int k, double[] pattern)
  {
    double output_net = 0;
    for (int j = 0; j < J; j++)
    {
      double hidden_net = outputForJthHiddenUnitValue(j, pattern);
      output_net += w_weights[k, j] * sigmoidFunction(hidden_net);
    }

    // add weighted bias to net
    output_net += w_weights[k, J] * bias;
    return tanhFunction(output_net);
  }

  private double outputForJthHiddenUnitValue(int j, double[] pattern)
  {
    double output_net = 0;
    for (int i = 0; i < I; i++)
    {
      output_net += v_weights[j, i] * pattern[i];
    }

    // add weighted bias to net
    output_net += v_weights[j, I] * bias;
    return sigmoidFunction(output_net);
  }

  private double tanhFunction(double x)
  {
    return (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x));
  }

  private double sigmoidFunction(double x)
  {
    return 1.0 / (1.0 + Math.Exp(-x));
  }
}