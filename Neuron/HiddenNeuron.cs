using Neuron.Tools;


namespace Neuron
{
    internal class HiddenNeuron : Neuron, ILearnable
    {
        public HiddenNeuron(int inputCount)
            : base(inputCount)
        {
            Weights = new double[inputCount];
            Inputs = new double[inputCount];

            InitWeightsByRandomValues(inputCount);
        }

        public override string NeuronType { get; } = "Hidden";

        public double[] Inputs { get; }

        public override void SetWeights(params double[] weights)
        {
            if (weights.Length != Weights.Length)
                throw new Exception("This is impossible to create such a hidden neuron!");

            Array.Copy(weights, Weights, weights.Length);
        }

        public void Learn(double error, double learningRate)
        {
            Delta = error * new SigmoidDx().GetResultBy(Output);
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

        public override double FeedForward(double[] inputs)
        {
            Array.Copy(inputs, Inputs, inputs.Length);

            if (inputs.Length != Weights.Length)
                throw new Exception("The number of input signals is not equal to the number of inputs to the hidden neuron!");

            var sum = Weights.Select((t, index) => inputs[index] * t).Sum();

            Output = new Sigmoid().GetResultBy(sum);

            return Output;
        }
    }
}