using System;

using Microsoft.Quantum.Simulation.Core;
using Microsoft.Quantum.Simulation.Simulators;

//Change namespace to desired algorithm
namespace dj_algorithm
{
    class Driver
    {
        static void Main(string[] args)
        {
            using (var qsim = new QuantumSimulator())
            {
                var watch = System.Diagnostics.Stopwatch.StartNew();

                //Uncomment the following for Deutsch Josze

                System.Console.WriteLine("Result for Deutsch Josze Algorithm:");
                Run_DeutschJozsa_Algorithm.Run(qsim).Wait();

                //Uncomment the following for Bernstein Vazirani
                /* 
                int n = 10;
                long[] a = new long[n];
                Random r = new Random();
                for(int i = 0; i < n; i++) {
                    a[i] = r.Next(0, 2);
                }
                var res = Run_BernsteinVazirani_Algorithm.Run(qsim, new QArray<long>(a)).Result;
                if(res.Equals("Success!")) {
                    System.Console.WriteLine("Result for Bernstein Vazirani Algorithm:");
                    for(int i = 0; i < n; i++) {
                        System.Console.Write(a[i] + " ");
                    }
                    System.Console.WriteLine();
                } else {
                    System.Console.WriteLine("incorrect");
                }
                */

                //Uncomment the following for Simon's
                /* 
                var res = Run_Simon_Algorithm.Run(qsim).Result;
                System.Console.WriteLine("Result for Simon's Algorithm:");
                System.Console.WriteLine(res);
                */


                //Uncomment the following for Grover's
                /* 
                var res = Run_Grovers_Algorithm.Run(qsim).Result;
                System.Console.WriteLine("Result for Grover's Algorithm:");
                System.Console.WriteLine(res);
                */


                watch.Stop();
                double totalTime = watch.ElapsedMilliseconds / 1000.0;
                System.Console.WriteLine("Time elapsed: " + totalTime);
            }
        }
    }
}