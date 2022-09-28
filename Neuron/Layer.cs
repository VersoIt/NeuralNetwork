using System.Collections;


namespace Neuron
{
    public class Layer : IEnumerable
    {
        public Layer(Neuron[] neurons)
        {
            Neurons = neurons;
        }

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

        public override string ToString() => $"{Neurons.First().NeuronType} layer";

        public IEnumerator GetEnumerator() => Neurons.GetEnumerator();
    }
}
