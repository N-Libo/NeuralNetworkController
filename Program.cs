class Program
{
  static void Main(string[] args)
  {
    // these matrices will be created from the genes of the GA
    double[,] v_weights = new double[NeuralNetwork.J, NeuralNetwork.I + 1];
    double[,] w_weights = new double[NeuralNetwork.K, NeuralNetwork.J + 1];

    randomizeWeights(ref v_weights);
    randomizeWeights(ref w_weights);

    NeuralNetwork nn = new NeuralNetwork(v_weights, w_weights);
    double[] predictions1 = nn.predict(new double[] { 0.1, 0.5, 0, 0 });
    double[] predictions2 = nn.predict(new double[] { 0.4, 0.1, 43, 25 });
    double[] predictions3 = nn.predict(new double[] { 0.9, 0.13, 87, 12 });
    double[] predictions4 = nn.predict(new double[] { 1.0, 2, 90, 15 });

    System.Console.WriteLine(string.Join(",", predictions1.ToList<double>()));
    System.Console.WriteLine(string.Join(",", predictions2.ToList<double>()));
    System.Console.WriteLine(string.Join(",", predictions3.ToList<double>()));
    System.Console.WriteLine(string.Join(",", predictions4.ToList<double>()));
  }

  static void randomizeWeights(ref double[,] weights)
  {
    int rows = weights.GetLength(0), cols = weights.GetLength(1);
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
        weights[i, j] = new Random().Next(0, 10);
  }
}