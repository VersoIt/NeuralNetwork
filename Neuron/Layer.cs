using System.Collections;
using System.Globalization;

namespace Neuron
{
    public class Layer : IEnumerable
    {

        public Layer(Neuron[] neurons, NeuronTypes type)
        {
            if (!neurons.All(x => x.NeuronType == type))
                throw new Exception("Невозможно создать нейроны разных типов в одном слое.");

            Neurons = neurons;
            NeuronType = type;
        }

        public NeuronTypes NeuronType { get; }

        public Neuron[] Neurons { get; }

        public Neuron this[int index]
        {
            get => Neurons[index];
            set => Neurons[index] = value;
        }

        public double[] GetSignals()
        {
            var signals = new double[Neurons.Count()];

            for (int index = 0; index < signals.Length; ++index)
                signals[index] = Neurons[index].Output;

            return signals;
        }

        public override string ToString() => NeuronType.ToString();

        public IEnumerator GetEnumerator() => Neurons.GetEnumerator();
    }
}
