using Neuron.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuron
{
    public class Neuron
    {

        public Neuron(int inputCount, NeuronTypes type = NeuronTypes.Normal)
        {
            Weights = new double[inputCount];
            NeuronType = type;

            Array.Fill(Weights, 1);
        }

        public double[] Weights { get; }

        public NeuronTypes NeuronType { get; }

        public double Output { get; private set; }

        public double FeedForward(double[] inputs)
        {
            if (inputs.Length != Weights.Length)
                throw new Exception("Число входных данных нейрона не равно числу входов в нейрон!");

            double sum = 0;

            for (int index = 0; index < Weights.Length; ++index)
                sum += inputs[index] * Weights[index];

            return new Sigmoida().GetResultBy(sum);
        }

        public override string ToString() => Output.ToString();

    }
}
