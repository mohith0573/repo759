#include <iostream>
#include <omp.h>

long factorial(int n)
{
    long f=1;
    for(int i=1;i<=n;i++)
        f*=i;
    return f;
}

int main()
{

#pragma omp parallel num_threads(4)
{
    int id = omp_get_thread_num();

#pragma omp single
    std::cout<<"Number of threads: "<<omp_get_num_threads()<<std::endl;

    std::cout<<"I am thread No. "<<id<<std::endl;
}

#pragma omp parallel for num_threads(4)
for(int i=1;i<=8;i++)
{
    long f = factorial(i);

#pragma omp critical
    std::cout<<i<<"!="<<f<<std::endl;
}

}
