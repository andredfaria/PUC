package projetoweka;

import java.util.Random;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Attribute;

public class weka {

    private String path;

    public weka(String path) {
        this.path = path;

    }
    private Instances dados;
    
    public void leDados() throws Exception {
        DataSource fonte = new DataSource(path);
        dados = fonte.getDataSet();
        if (dados.classIndex() == -1) {
            dados.setClassIndex(dados.numAttributes() - 1);
        }
    }

    public void imprimeDados() {
        for (int i = 0; i < dados.numInstances(); i++) {
            Instance atual = dados.instance(i);
            System.out.println((i + 1) + ": " + atual + "\n");
        }
    }

    public void arvoreDeDecisaoJ48() throws Exception {
        J48 tree = new J48();
        tree.buildClassifier(dados);
        System.out.println(tree);
        System.out.println("Avaliacao inicial: \n");
        Evaluation avaliacao;
        avaliacao = new Evaluation(dados);
        avaliacao.evaluateModel(tree, dados);
        System.out.println("--> Instancias corretas: "
                + avaliacao.correct() + "\n");
        System.out.println("Avaliacao cruzada: \n");
        Evaluation avalCruzada;
        avalCruzada = new Evaluation(dados);
        avalCruzada.crossValidateModel(tree, dados, 10, new Random(1));
        System.out.println("--> Instancias corretas CV: "
                + avalCruzada.correct() + "\n");
    }

    public void InstanceBased() throws Exception {
        IBk k3 = new IBk(3);
        k3.buildClassifier(dados);
        Instance newInst = dados.instance(5);
        newInst.setDataset(dados);
        newInst.setValue(0, 7.2);
        newInst.setValue(1, 3.5);
        newInst.setValue(2, 8.2);
        newInst.setValue(3, 5.2);
        double pred = k3.classifyInstance(newInst);
        System.out.println("Predição: " + pred);
        Attribute a = dados.attribute(4);
        String predClass = a.value((int) pred);
        System.out.println("Predição: " + predClass);

    }
}
