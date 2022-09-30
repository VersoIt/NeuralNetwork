using NeuralNetwork.Tools;
using NeuralNetwork.Abstract;


namespace NeuralNetwork
{
    public class OutputNeuron : Neuron
    {

        public OutputNeuron(int inputCount)
            : base(inputCount)
        {
            Weights = new double[inputCount];
            Inputs = new double[inputCount];

            InitWeightsByRandomValues(inputCount);
        }

        public override string NeuronType { get; } = "Output";

        public double[] Inputs { get; }

        public override void SetWeights(params double[] weights)
        {
            if (weights.Length != Weights.Length)
                throw new Exception("This is impossible to create such a output neuron!");

            Array.Copy(weights, 0, Weights, 0, weights.Length);
        }

        public override void Learn(double error, double learningRate)
        {
            Delta = error * MathFunction.GetResultOfSigmoidDx(Output);
            for (var i = 0; i < Weights.Length; ++i)
            {
                var weight = Weights[i];
                var input = Inputs[i];

                var newWeight = weight - input * Delta * learningRate;

                Weights[i] = newWeight;
            }
        }

        private void InitWeightsByRandomValues(int inputCount)
        {
            var random = new Random();

            for (var i = 0; i < inputCount; ++i)
            {
                Weights[i] = random.NextDouble();
                Inputs[i] = 0.0;
            }
        }

        public override double FeedForward(IReadOnlyList<double> inputs)
        {
            Array.Copy(inputs.ToArray(), Inputs, inputs.Count());

            if (inputs.Count() != Weights.Length)
                throw new Exception("The number of input signals is not equal to the number of inputs to the output neuron!");

            var sum = Weights.Select((w, index) => inputs[index] * w).Sum();

            Output = MathFunction.GetResultOfSigmoid(sum);

            return Output;
        }
    }
}
