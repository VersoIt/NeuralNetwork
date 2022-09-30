using System.Threading;

namespace ThreadingTest
{
    internal class Program
    {
        private static void Test()
        {
            List<Thread> threads = new List<Thread>();
            for (int i = 0; i < 10; ++i)
            {
                Thread thread = new Thread(() =>
                {
                    Thread.Sleep(1000);
                    Console.WriteLine("Thread");
                });
                thread.Start();
                threads.Add(thread);
            }
            foreach(var thread in threads)
            {
                thread.Join();
            }
            Console.WriteLine("Test");
        }
        static void Main(string[] args)
        {
            Test();
            Console.WriteLine("Main");
        }
    }
}