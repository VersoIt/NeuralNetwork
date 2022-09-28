using System.Globalization;
using System.Linq;
using Neuron.Tools;


namespace Neuron
{
    public class DefaultNeuron : Neuron
    {
        public DefaultNeuron(int inputCount, NeuronTypes type = NeuronTypes.Normal)
            : base(inputCount, type)
        {
            Weights = new double[inputCount];
            Inputs = new double[inputCount];

            InitWeightsByRandomValues(inputCount);
        }

        public double[] Inputs { get; }

        public override void SetWeights(params double[] weights)
        {
            if (weights.Length != Weights.Length)
                throw new Exception("This is impossible to create such a neuron!");

            Array.Copy(weights, Weights, weights.Length);
        }

        public override void Learn(double error, double learningRate)
        {
            if (NeuronType == NeuronTypes.Input)
                return;

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
                if (NeuronType == NeuronTypes.Input)
                {
                    Weights[i] = 1;
                }
                else
                {
                    Weights[i] = random.NextDouble();
                }
                Inputs[i] = 0.0;
            }
        }

        public override double FeedForward(double[] inputs)
        {
            Array.Copy(inputs, Inputs, inputs.Length);

            if (inputs.Length != Weights.Length)
                throw new Exception("The number of inputs to the neuron is not equal to the number of inputs to the neuron!");

            var sum = Weights.Select((t, index) => inputs[index] * t).Sum();

            Output = NeuronType != NeuronTypes.Input ? new Sigmoid().GetResultBy(sum) : sum;

            return Output;
        }

        public override string ToString() => Output.ToString(CultureInfo.InvariantCulture);
    }
}
