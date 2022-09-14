using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Neuron.Tests
{
    [TestClass()]
    public class NeuralNetworkTests
    {
        [TestMethod()]
        public void FeedForwardTest()
        {
            var topology = new Topology(4, 1, 2);
            var network = new NeuralNetwork(topology);

            network[0]
        }
    }
}