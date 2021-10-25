/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.matrix;

import com.mycompany.neuralnetwork.exceptions.MatrixException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
/**
 *
 * @author dmytr
 */
public class Matrix {
    private int axes;
    private int[] shape;
    private int size;
    private List<Number> elements;
    private Class elType=Double.class;
    
    public Matrix(int[] shape){
        this.shape=shape;
        axes=shape.length; 
        this.size=Arrays.stream(shape).reduce((acc,x)->acc*x).orElse(1);
        this.elType=Integer.class;
        try{
            elements=Matrices.fillMatrix(Integer.class,size);
        } catch(Exception ex){
            System.out.println(ex.getMessage());
        }
    }
    
    public <T extends Number> Matrix(int[] shape, Class<T> type) throws MatrixException{
        this.shape=shape;
        axes=shape.length; 
        this.size=Arrays.stream(shape).reduce((acc,x)->acc*x).orElse(1);
        this.elType=type;
        elements=Matrices.fillMatrix(type,size);
    }
    
    public <T extends Number> Matrix(List<Number> elements, int[] shape, Class<T> type) throws MatrixException{
        this.shape=shape;
        axes=shape.length; 
        this.size=Arrays.stream(shape).reduce((acc,x)->acc*x).orElse(1);
        this.elType=type;
        this.elements=elements.stream().map(e->Matrices.turnToElType(type, e.doubleValue())).collect(Collectors.toList());
    }
    
    public static <T extends Number> Matrix zeros(int[] shape, Class<T> type) throws MatrixException{
        return new Matrix(IntStream.iterate(0, i->i=0)
                .limit(Arrays.stream(shape).reduce((acc,x)->acc*x).orElse(0))
                .boxed().map(i->Matrices.turnToElType(type, i)).collect(Collectors.toList()),shape,type);
    }
    
    public void matrixMult(Matrix matrix) throws MatrixException{
        if(axes>2 || matrix.getAxes()>2){
            throw new MatrixException("Only matrices with not more than 2 axes may be multiplied");
        }
        int this_columns=axes>1?shape[1]:1;
        int this_rows=shape[0];
        int matrix_columns=matrix.getAxes()==1?1:matrix.getShape()[1];
        int matrix_rows=matrix.getShape()[0];
        if(matrix_rows!=this_columns){
            throw new MatrixException("Matrices cannot be multiplied");
        }
        List<Number> newElems=new ArrayList<>();
        for(int i=0;i<this_rows;i++){            
            for(int j=0;j<matrix_columns;j++){
                double newElem=0d;
                for(int k=j;k<matrix.getSize();k+=matrix_columns){
                    int currentPos=k==j?((size/this_rows)*i):(k/matrix_columns+(size/this_rows)*i);
                    newElem+=elements.get(currentPos).doubleValue()*
                            matrix.getElements().get(k).doubleValue();
                }
                newElems.add(Matrices.turnToElType(elType, newElem));
            }
        }
        shape=matrix_columns==1?new int[]{shape[0]}:new int[]{shape[0],matrix_columns};
        modifySelf(new Matrix(newElems,shape,elType));
    }
    
    public void add(Double num){ 
        elements=(List<Number>) elements.stream().map(e->Matrices.turnToElType(elType, e.doubleValue()+num))
                .collect(Collectors.toList());
    }
    
    public void add(Matrix matrix) throws MatrixException{
        Matrices.checkMatrixesForAddition(this, matrix);
        for(int i=0;i<elements.size();i++){
            elements.set(i, Matrices.turnToElType(elType,
                    elements.get(i).doubleValue()+matrix.elements.get(i).doubleValue()));
        }
    }
    
    public void subtract(Double num){        
        elements=(List<Number>) elements.stream().map(e->Matrices.turnToElType(elType, e.doubleValue()-num))
                .collect(Collectors.toList());
    }
    
    public void subtract(Matrix matrix) throws MatrixException{
        Matrices.checkMatrixesForAddition(this, matrix);
        for(int i=0;i<elements.size();i++){
            elements.set(i, Matrices.turnToElType(elType,
                    elements.get(i).doubleValue()-matrix.elements.get(i).doubleValue()));
        }
    }
    
    public void multiply(Double num){        
        elements=(List<Number>) elements.stream().map(e->Matrices.turnToElType(elType, e.doubleValue()*num))
                .collect(Collectors.toList());
    }
    
    public void multiply(Matrix matrix) throws MatrixException{
        Matrices.checkMatrixesForMultiplication(this, matrix);
        if(matrix.size>size){
            Matrix newMatrix=Matrices.copyOf(matrix);
            newMatrix.multiply(this);
            modifySelf(newMatrix);
        } else{
            int sizeDiff=size/matrix.size;
            for(int i=0,j=0;i<elements.size();i++){
                if(i>= sizeDiff && i%sizeDiff==0)j++;
                elements.set(i,Matrices.turnToElType(
                        elType,elements.get(i).doubleValue()*matrix.elements.get(j).doubleValue()));
            }
        }
    }
    
    public void divide(Double num){        
        elements=(List<Number>) elements.stream().map(e->Matrices.turnToElType(elType, e.doubleValue()/num))
                .collect(Collectors.toList());
    }
    
    public void divide(Matrix matrix) throws MatrixException{
        Matrices.checkMatrixesForMultiplication(this, matrix);
        if(matrix.size>size){
            Matrix newMatrix=Matrices.copyOf(matrix);
            newMatrix.divide(this);
            modifySelf(newMatrix);
        } else{
            Matrices.checkMatrixesForMultiplication(this, matrix);        
            int sizeDiff=size/matrix.size;
            for(int i=0,j=0;i<elements.size();i++){
                if(i>= sizeDiff && i%sizeDiff==0)j++;
                elements.set(i,Matrices.turnToElType(elType,
                        elements.get(i).doubleValue()/matrix.elements.get(j).doubleValue()));            
            }
        }
    }
    
    public Number sum(){
        return elements.stream().map(n->n.doubleValue()).reduce((acc,x)->acc+x).orElse(0d);
    }
    
    public Matrix sum(int axis) throws MatrixException, Exception{
        long beginPoint=System.currentTimeMillis();
        int[] newShape=new int[shape.length-1];
        int step=size;
        for(int i=0,j=0;i<shape.length;i++){
          if(i!=axis){
              newShape[j]=shape[i];
              j++;
          } 
          if(i<=axis)step/=shape[i];
        }
        Matrix newMatrix=new Matrix(newShape,elType);
        List<Number> oldElements=new ArrayList<>(elements);
        List<Number> newElements=new ArrayList<>();
        List<List<Number>> subLists=new ArrayList<>();        
        while(!oldElements.isEmpty()){
            subLists.add(oldElements.stream().limit(step).collect(Collectors.toList()));
            oldElements=oldElements.stream().skip(step).collect(Collectors.toList());
            if(subLists.size()==shape[axis]){
                newElements.addAll(Matrices.combineElements(subLists));
                subLists.clear();
            }
        }
        newMatrix.setElements(newElements);
        System.out.println("Sum:"+(System.currentTimeMillis()-beginPoint));
        return newMatrix;
    }
    
    public int argmax(){
        return elements.stream().map(e->e.doubleValue()).collect(Collectors.toList())
                .indexOf(elements.stream().map(e->e.doubleValue()).max(Double::compare).get());
    }
    
    public void transpose() throws MatrixException{
        if(axes!=2)throw new MatrixException("Matrix should have 2 axes to be trasposed.");
        Number[] elemsAsArrT=elements.toArray(new Number[size]);
        elements=new ArrayList<>();
        for(int i=0;i<shape[1];i++){
            for(int j=i;j<size;j+=shape[1]){
                elements.add(elemsAsArrT[j]);
            }
        }
        shape=new int[]{shape[1],shape[0]};        
    }
    
    public void applyFunc(UnaryOperator<Double> func){
        elements=(List<Number>) elements.stream().map(
                e->Matrices.turnToElType(elType,func.apply(e.doubleValue()))).collect(Collectors.toList());
    }
    
    private void modifySelf(Matrix matrix){
        this.shape=matrix.shape;
        this.elType=matrix.elType;
        this.elements=matrix.elements;
        this.size=matrix.size;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final Matrix other = (Matrix) obj;
        if (!Arrays.equals(this.shape, other.shape)) {
            return false;
        }
        if (!Objects.equals(this.elements, other.elements)) {
            return false;
        }
        if (!Objects.equals(this.elType, other.elType)) {
            return false;
        }
        return true;
    }
    
        
    public String toString(){
        return Matrices.matrixToString(this);
    }

    public List<Number> getElements() {
        return elements;
    }

    public int getAxes() {
        return axes;
    }

    public int[] getShape() {
        return shape;
    }

    public int getSize() {
        return size;
    }

    public Class getElType() {
        return elType;
    }

    public void setElements(List<Number> elements) {
        this.elements = elements;
    }

    public void setShape(int[] shape) {
        this.shape = shape;
        axes= this.shape.length;
    } 

    public void setElType(Class elType) {
        this.elType = elType;
    }
        
}
