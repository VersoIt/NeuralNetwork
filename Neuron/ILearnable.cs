namespace Neuron
{
    internal interface ILearnable
    {
        public void Learn(double error, double learningRate);
    }
}
