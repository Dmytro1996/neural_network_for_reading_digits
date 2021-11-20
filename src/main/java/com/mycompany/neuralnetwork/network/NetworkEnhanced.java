/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.network;

import com.google.gson.Gson;
import com.mycompany.neuralnetwork.cost.Cost;
import com.mycompany.neuralnetwork.cost.CrossEntropyCost;
import com.mycompany.neuralnetwork.neuron.Neuron;
import java.io.BufferedReader;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.ops.NDMath;
import org.nd4j.linalg.ops.transforms.Transforms;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

/**
 *
 * @author dmytr
 */
public class NetworkEnhanced {
    
    private int num_layers;
    private int[] shape;
    private List<INDArray> weights;
    private List<INDArray> biases;
    private Cost cost;
    private Neuron neuron;
    
    public NetworkEnhanced(int[] shape, Cost cost, Neuron neuron){
        this.shape=shape;
        this.num_layers=shape.length;
        weights=new LinkedList<>();
        biases=new LinkedList<>();
        this.cost=cost;
        this.neuron=neuron;
        default_weight_initializer();
    }
    
    public INDArray feedforward(INDArray activation){
        for(int i=0;i<weights.size();i++){
            activation=neuron.fun(weights.get(i).mmul(activation).add(biases.get(i)));
        }
        return activation;
    }
    
    /*public void SGD(List<INDArray[]> train_data, int epochs, int mini_batch_size, double eta,
            double lambda, boolean monitor_eval_cost, boolean monitor_train_cost,
            boolean monitor_train_acc, List<INDArray[]>...eval_data){
        int len_eval=(eval_data.length!=0)?eval_data[0].size():0;
        
        int len_train=train_data.size();
        for(int i=0;i<epochs;i++){
            Collections.shuffle(train_data);
            List<List<INDArray[]>> mini_batches=IntStream.range(0,(len_train/mini_batch_size))
                    .boxed().map(c->train_data.subList(c*mini_batch_size, (c+1)*mini_batch_size))
                    .collect(Collectors.toList());
            System.out.println(mini_batches.size());
            long beginPoint=System.currentTimeMillis();
            mini_batches.forEach(mini_batch->{
                    update_mini_batch(mini_batch,eta,lambda,train_data.size());
            });
            System.out.println("\n"+(System.currentTimeMillis()-beginPoint));
            if(monitor_train_cost){
                System.out.println("Cost on training data: {"+countTotalCost(train_data, lambda)+"}");
            }
            if(monitor_train_acc){
                System.out.println("Accuracy on training data: {"+countAccuracy(train_data, true)+"} // {"+len_train+"}");
            }
            if(monitor_eval_cost){
                System.out.println("Cost on evaluation data: {"+countTotalCost(eval_data[0], lambda)+"}");
            }
            if(len_eval>0){
                System.out.println("Epoch {"+i+"}: {"+countAccuracy(eval_data[0],false)+"} // {"+len_eval+"}");
            }
            else{
                System.out.println("Epoch {"+i+"}: completed");
            }
        }
    }
    
    public void update_mini_batch(List<INDArray[]> mini_batch, double eta, double lambda, int lenTrainData){
        INDArray[][] nablas=mini_batch.parallelStream().map(mb->backprop(mb[0],mb[1]))
                .reduce((acc,x)->{
            //System.out.println("acc biases:"+acc[0][1]);
            //System.out.println("x biases:"+x[0][1]);
            //System.out.println("acc weights:"+acc[1][1]);
            //System.out.println("x weights:"+x[1][1]);
            for(int i=0;i<num_layers-1;i++){                
                acc[0][i]=acc[0][i].add(x[0][i]);
                acc[1][i]=acc[1][i].add(x[1][i]);
            }
            return acc;
        }).get(); 
        for(int i=0;i<shape.length-1;i++){
            //System.out.println(eta/mini_batch.size());
            nablas[1][i]=nablas[1][i].mul(eta/mini_batch.size());
            nablas[0][i]=nablas[0][i].mul(eta/mini_batch.size());
            //System.out.println(1-eta*Math.round(lambda/lenTrainData));
            //System.out.println(nablas[1][i]);
            //System.out.println("Params:"+eta+" "+lambda+" "+lenTrainData);
            //System.out.println("Weights mul:"+(1-eta*(lambda/lenTrainData)));
            weights.set(i, weights.get(i).mul(1d-eta*Math.round(lambda/lenTrainData)).sub(nablas[1][i]));
            //System.out.println("weights:"+weights.get(i));
            biases.set(i,biases.get(i).sub(nablas[0][i]));
            //System.out.println("Biases:"+biases.get(i));
        }
        System.out.print("-");
    }
    
    public INDArray[][] backprop(INDArray x,INDArray y){
        INDArray[] nabla_b=new INDArray[num_layers-1];
        INDArray [] nabla_w=new INDArray[num_layers-1];
        INDArray activation=x;
        List<INDArray> activations=new LinkedList<>();
        activations.add(x);
        List<INDArray> zs=new LinkedList<>();
        for(int i=0;i<weights.size();i++){
            //System.out.println(weights.get(i));
            //System.out.println(biases.get(i));
            zs.add(weights.get(i).mmul(activation).add(biases.get(i)));
            //System.out.println("z:"+zs.get(i));
            activation=neuron.fun(zs.get(i));
            //System.out.println("Activation:"+activation);
            activations.add(activation);
        }
        //System.out.println("Activations:"+activations.get(activations.size()-1));
        INDArray delta=cost.delta(activations.get(activations.size()-1),y, null);
        //System.out.println("delta:"+delta);
        nabla_b[nabla_b.length-1]=delta.dup();
        //System.out.println(nabla_b[nabla_b.length-1]==delta);
        //System.out.println("nabla_b[nabla_b.length-1]:"+nabla_b[nabla_b.length-1]);
        nabla_w[nabla_w.length-1]=delta.reshape(new int[]{(int)delta.shape()[0],1})
                .mmul(activations.get(activations.size()-2).reshape(
                        new int[]{1,(int)activations.get(activations.size()-2).shape()[0]}));
        //System.out.println("nabla_w[nabla_w.length-1]:"+nabla_w[nabla_w.length-1]);
        for(int i=2;i<num_layers;i++){
            int j=i;
            INDArray z=zs.get(zs.size()-i);
            INDArray sp=neuron.derivative(z);
            delta=weights.get(weights.size()-i+1).transpose().mmul(delta).mul(sp);
            //System.out.println("delta"+i+":"+delta);
            nabla_b[nabla_b.length-i]=delta.dup();
            //System.out.println("nabla_b[nabla_b.length-i]:"+nabla_b[nabla_b.length-i]);
            nabla_w[nabla_w.length-i]=delta.reshape(new int[]{(int)delta.shape()[0],1})
                .mmul(activations.get(activations.size()-j-1).reshape(
                        new int[]{1,(int)activations.get(activations.size()-j-1).shape()[0]}));
            //System.out.println("nabla_w[nabla_w.length-i]:"+nabla_w[nabla_w.length-i]);
        }
        return new INDArray[][]{nabla_b,nabla_w};
    }*/
    
    public long countAccuracy(List<INDArray[]> eval_data, boolean convert){
        if(convert){
        return eval_data.parallelStream().filter(c->
                feedforward(c[0]).argMax(0).data().asInt()[0]==c[1].argMax(0).data().asInt()[0])
                .count();
        }
        return eval_data.parallelStream().filter(c->
                feedforward(c[0]).argMax(0).data().asInt()[0]==c[1].data().asInt()[0])
                .count();
    }
    
    public double countTotalCost(List<INDArray[]> data, double lambda){
        double weightsSquare=weights.parallelStream().map(w->new NDMath().square(w).sum().data().asDouble()[0])
                .reduce((acc,x)->acc+x).orElse(0d);
        return data.parallelStream().peek((arr)->{
            if(arr[1].data().length()==1)arr[1]=vectorized_result(arr[1]);
                }).map(arr->cost.fun(feedforward(arr[0]), arr[1])).reduce((acc,x)->acc+x).orElse(0d)+
                0.5*(lambda/data.size())*weightsSquare;
    }
    
    public INDArray vectorized_result(INDArray expected_output){
        return Nd4j.zeros(shape[num_layers-1]).putScalar(new int[]{(int)expected_output.data().asDouble()[0]},1d);
    }
    
    public void large_weight_initializer(){
        Random rand=new Random();
        for(int i=0;i<num_layers;i++){
            if(i<num_layers-1)weights.add(Nd4j.create(DoubleStream.generate(()->rand.nextGaussian())
                    .limit(shape[i+1]*shape[i]).toArray(),
                    new long[]{shape[i+1],shape[i]},DataType.DOUBLE));
            if(i>0)
                biases.add(Nd4j.create(DoubleStream.generate(()->rand.nextGaussian())
                        .limit(shape[i]).toArray(),new long[]{shape[i]},DataType.DOUBLE));
        }
    }
    
    public void default_weight_initializer(){
        Random rand=new Random();
        for(int i=0;i<num_layers;i++){
            int j=i;
            if(i<num_layers-1)weights.add(Nd4j.create(DoubleStream.generate(()->rand.nextGaussian()/Math.sqrt(shape[j]))
                    .limit(shape[i+1]*shape[i]).toArray(),
                    new long[]{shape[i+1],shape[i]},DataType.DOUBLE));
            if(i>0)
                biases.add(Nd4j.create(DoubleStream.generate(()->rand.nextGaussian())
                        .limit(shape[i]).toArray(),new long[]{shape[i]},DataType.DOUBLE));
        }
    }
    
    public void save(String fileName) throws IOException{
        BufferedWriter br=new BufferedWriter(new FileWriter(fileName, true));
        Gson gson=new Gson();
        br.write(gson.toJson(this, NetworkEnhanced.class));
        br.close();
    }
    
    public NetworkEnhanced load(String fileName) throws IOException{
        FileReader fr=new FileReader(fileName);
        BufferedReader br=new BufferedReader(fr);
        StringBuilder text=new StringBuilder();
        String s="";
        while((s=br.readLine())!=null){
            text.append(s);
        }
        NetworkEnhanced loadedNet=new Gson().fromJson(text.toString(), NetworkEnhanced.class);
        this.weights=loadedNet.getWeights();
        this.biases=loadedNet.getBiases();
        return loadedNet;
    }

    public List<INDArray> getWeights() {
        return weights;
    }

    public List<INDArray> getBiases() {
        return biases;
    }

    public int getNum_layers() {
        return num_layers;
    }

    public int[] getShape() {
        return shape;
    }

    public Cost getCost() {
        return cost;
    }

    public Neuron getNeuron() {
        return neuron;
    }
    

    public void setWeights(List<INDArray> weights) {
        this.weights = weights;
    }

    public void setBiases(List<INDArray> biases) {
        this.biases = biases;
    }
}
