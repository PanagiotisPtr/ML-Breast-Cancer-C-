#include <iostream>
#include <vector>
#include <cstdio>
#include <time.h>
#include <algorithm>
#include <math.h>

using namespace std;

class Adaline
{
public:
	Adaline(float h, int e);
	void train(vector< vector<float> > const &X, vector<int> const &y);
	int predict(vector<float> const &X);
	void test(vector< vector<float> > const &X, vector<int> const &y);
private:
	float netInput(vector<float> const &X);
	float bias = 1;
	vector<float> m_w;
	float eta;
	float epochs;
};

Adaline::Adaline(float h, int e)
{
	eta = (float)h;
	epochs = (int)e;
}

void Adaline::train(vector< vector<float> > const &X, vector<int> const &y)
{
	m_w = vector<float>(X[0].size());
	vector<int> m_err;
	cout << "Started Training" << endl;
	for(int i = 0; i < epochs; i++)
	{
		int errors = 0;
		for(int j = 0; j < X.size(); j++)
		{
			float update = eta * (y[j] - predict(X[j]));
			//m_w += X[j] * update;
			for(int q = 0; q < m_w.size(); q++)m_w[q]+=X[j][q]*update;
			bias += update;
			errors += update!=0 ? 1 : 0;
		}
		printf("Errors at Epoch %d: %d", i+1, errors);
		printf("\t Accuracy: %0.2f\n", (float)(100-errors)/100);
		//Adaptive Learning Rate
		if(errors==0)break;
		m_err.push_back(errors);
	}
}

void Adaline::test(vector< vector<float> > const &X, vector<int> const &y)
{
	int error = 0;
	for(int i = 0; i < X.size(); i++)
	{
		int ans = predict(X[i]);
		if(ans!=y[i])error++;
	}
	cout << "Accuracy on Test: " << (float)(199-error)/199 << endl;
}

float Adaline::netInput(vector<float> const &X)
{
	float sum = bias;
	for(int i = 0; i < X.size(); i++){
		sum += m_w[i]*X[i];
	}
	return tanh(sum);
	//return dotProduct(m_w,X) + bias;
}

int Adaline::predict(vector<float> const &X)
{
	return netInput(X)>=0 ? 1 : -1;
}

//////////////////////// DATA ////////////////////////

void loadIris(vector< vector<float> > &X, vector<int> &y)
{
	srand(time(NULL));
	freopen("iris.train", "rt", stdin);
	X = vector< vector<float> >(100, vector<float>(4));
	y = vector<int>(100);
	vector< vector<float> > temp(100, vector<float>(5));
	for(int i = 0; i < 100; i++)for(int j = 0; j < 5; j++)cin >> temp[i][j];	
	random_shuffle(temp.begin(), temp.end());
	for(int i = 0; i < 100; i++)
	{
		for(int j = 0; j < 4; j++)X[i][j] = temp[i][j];
		y[i] = temp[i][4];
	}
}

void loadBC(vector< vector<float> > &X, vector<int> &y,vector< vector<float> > &test_X, vector<int> &test_y)
{
	srand(time(NULL));
	freopen("BC.train", "rt", stdin);
	X = vector< vector<float> >(200, vector<float>(9));
	y = vector<int>(200);
	vector< vector<float> > temp;

	for(int i = 0; i < 699; i++)
	{
		temp.push_back(vector<float>());
		for(int j = 0; j < 11; j++)
		{
			int a;
			cin >> a;
			temp.back().push_back(a);
		}
		if(temp.back().back()==4)temp.back().back() = 1;
		if(temp.back().back()==2)temp.back().back() = -1;
	}

	for(int i = 0; i < 699; i++)temp[i].erase(temp[i].begin());

	for(int i = 0; i < 200; i++)
	{
		for(int j = 0; j < 9; j++)
		{
			X[i][j] = temp[i][j];
			//cout << X[i][j] << " ";
		}
		y[i] = temp[i][9];
		//cout << y[i] << endl;
	}

	test_X = vector< vector<float> >(499, vector<float>(9));
	test_y = vector<int>(499);

	for(int i = 0; i < 499; i++)
	{
		for(int j = 0; j < 9; j++)
		{
			test_X[i][j] = temp[i+200][j];
			//cout << X[i][j] << " ";
		}
		test_y[i] = temp[i+200][9];
		//cout << y[i] << endl;
	}
}



int main()
{
	Adaline model(0.01, 100);
	vector< vector<float> > X;
	vector<int> y;
	vector< vector<float> > test_X;
	vector<int> test_y;
	loadBC(X,y,test_X,test_y);
	model.train(X,y);
	model.test(test_X,test_y);
	return 0;
}
