package com.example.xor;

import com.example.tfcapi.AbstractClassifier.History;
import com.example.tfcapi.*;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.JFrame;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Rectangle;
import java.awt.geom.Ellipse2D;
import java.util.Random;

/**
 * Project 2 - The classic XOR problem, solved with a hidden layer.
 *
 * A single-layer perceptron cannot represent the XOR function because XOR
 * is not linearly separable. Adding one hidden layer with a non-linear
 * activation (Tanh) lets the network bend the decision surface so the two
 * classes become separable.
 *
 * This project:
 * 1. generates 400 noisy 2-D XOR points in [-1,1]^2 (seeded, ~8% label noise),
 * 2. trains a tiny MLP: Input(2) -> Dense(8) -> Tanh -> Dense(2),
 * 3. evaluates a 100x100 prediction grid over the plane,
 * 4. renders the colored decision boundary plus the training points
 *    as an interactive JFreeChart window titled "XOR Decision Boundary".
 */
public final class XorDecisionBoundary {

    private static final int NUM_POINTS = 400;   // noisy training samples
    private static final int GRID = 100;         // 100x100 grid for the surface
    private static final long SEED = 7L;
    private static final double NOISE_RATE = 0.08;

    public static void main(String[] args) throws Exception {
        boolean showGui = true;
        for (String a : args) {
            if ("--no-gui".equalsIgnoreCase(a)) showGui = false;   // headless runs / CI
        }

        // ---------- 1. Generate the noisy XOR dataset --------------------
        Random r = new Random(SEED);
        float[] x = new float[NUM_POINTS * 2];
        int[] y = new int[NUM_POINTS];
        for (int i = 0; i < NUM_POINTS; i++) {
            double px = r.nextDouble() * 2 - 1;   // uniform in [-1, 1]
            double py = r.nextDouble() * 2 - 1;
            x[i * 2] = (float) px;
            x[i * 2 + 1] = (float) py;
            // sign(px*py) > 0 means both coordinates share a sign -> class 1,
            // otherwise opposite signs -> class 0. This is exactly XOR.
            int label = (px * py > 0) ? 1 : 0;
            if (r.nextDouble() < NOISE_RATE) label ^= 1;   // ~8% random flips
            y[i] = label;
        }
        System.out.println("Generated " + NUM_POINTS + " noisy XOR points "
                + "(~" + Math.round(NOISE_RATE * 100) + "% label flips)");

        // ---------- 2. Small train/validation split ----------------------
        int[] order = seededOrder(NUM_POINTS, SEED);
        int trainN = 320, valN = NUM_POINTS - trainN;
        float[] trainX = new float[trainN * 2]; int[] trainY = new int[trainN];
        float[] valX = new float[valN * 2];     int[] valY = new int[valN];
        for (int i = 0; i < NUM_POINTS; i++) {
            if (i < trainN) { copyRow(x, y, order[i], trainX, trainY, i); }
            else            { copyRow(x, y, order[i], valX, valY, i - trainN); }
        }

        // ---------- 3. Build and train the network -----------------------
        try (Sequential net = new Sequential(Optimizer.adam(0.03f),
                new Input(2),
                new Dense(8), new Tanh(),
                new Dense(2))) {

            History h = net.fit(trainX, trainY, 300, 32, valX, valY, true);

            System.out.printf("Final: train loss %.4f | val loss %.4f | val acc %.3f%n",
                    h.finalTrainLoss(), h.finalValLoss(), h.finalValAcc());

            // ---------- 4. Accuracy on every generated point -------------
            float[] probs = net.predict(x);
            int correct = 0;
            for (int i = 0; i < NUM_POINTS; i++) {
                if (Metrics.argmax(probs, i * 2, 2) == y[i]) correct++;
            }
            System.out.printf("Accuracy on all %d points: %.3f (%d/%d)%n",
                    NUM_POINTS, (double) correct / NUM_POINTS, correct, NUM_POINTS);

            // ---------- 5. Decision boundary chart -----------------------
            // Evaluate the whole 100x100 grid in one forward pass while the
            // network is still open, then render it with JFreeChart.
            if (showGui) {
                float[] grid = new float[GRID * GRID * 2];
                int g = 0;
                for (int gy = 0; gy < GRID; gy++) {
                    double py = -1.0 + 2.0 * gy / (GRID - 1);
                    for (int gx = 0; gx < GRID; gx++) {
                        double px = -1.0 + 2.0 * gx / (GRID - 1);
                        grid[g * 2] = (float) px;
                        grid[g * 2 + 1] = (float) py;
                        g++;
                    }
                }
                float[] gridProbs = net.predict(grid);
                final float[] probsF = gridProbs;
                javax.swing.SwingUtilities.invokeLater(() -> showBoundary(probsF, x, y));
                System.out.println("Close the chart window to exit.");
            } else {
                System.out.println("(--no-gui: skipping chart)");
            }
        }
    }

    /* ------------------------- helper methods --------------------------- */

    /** Fisher-Yates shuffle of indices 0..n-1 with a fixed seed (reproducible). */
    static int[] seededOrder(int n, long seed) {
        int[] o = new int[n];
        for (int i = 0; i < n; i++) o[i] = i;
        Random r = new Random(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1), t = o[i]; o[i] = o[j]; o[j] = t;
        }
        return o;
    }

    static void copyRow(float[] src, int[] srcY, int row,
                        float[] dst, int[] dstY, int dstRow) {
        dst[dstRow * 2] = src[row * 2];
        dst[dstRow * 2 + 1] = src[row * 2 + 1];
        dstY[dstRow] = srcY[row];
    }

    /**
     * Renders the learned decision surface.
     *
     * Series 0/1 hold the 100x100 grid points colored by the network's
     * predicted class (the decision surface). Series 2/3 overlay the
     * original training points with larger, fully opaque markers so you can
     * see where the noisy samples fall relative to the boundary.
     */
    static void showBoundary(float[] gridProbs, float[] trainX, int[] trainY) {
        XYSeriesCollection dataset = new XYSeriesCollection();
        XYSeries grid0 = new XYSeries("predicted class 0", false);
        XYSeries grid1 = new XYSeries("predicted class 1", false);
        XYSeries pts0  = new XYSeries("noisy points, class 0", false);
        XYSeries pts1  = new XYSeries("noisy points, class 1", false);

        int g = 0;
        for (int gy = 0; gy < GRID; gy++) {
            double py = -1.0 + 2.0 * gy / (GRID - 1);
            for (int gx = 0; gx < GRID; gx++) {
                double px = -1.0 + 2.0 * gx / (GRID - 1);
                if (Metrics.argmax(gridProbs, g * 2, 2) == 0) {
                    grid0.add(px, py);
                } else {
                    grid1.add(px, py);
                }
                g++;
            }
        }
        for (int i = 0; i < trainY.length; i++) {
            double px = trainX[i * 2], py = trainX[i * 2 + 1];
            if (trainY[i] == 0) pts0.add(px, py); else pts1.add(px, py);
        }
        dataset.addSeries(grid0);
        dataset.addSeries(grid1);
        dataset.addSeries(pts0);
        dataset.addSeries(pts1);

        NumberAxis domain = new NumberAxis("x1");
        domain.setRange(-1.15, 1.15);
        NumberAxis range = new NumberAxis("x2");
        range.setRange(-1.15, 1.15);

        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer(false, true);
        // Decision surface cells: small squares, soft pastel fills.
        renderer.setSeriesPaint(0, new Color(120, 170, 255, 90));   // light blue
        renderer.setSeriesPaint(1, new Color(255, 140, 105, 90));   // light salmon
        renderer.setSeriesShape(0, new Rectangle(6, 6));
        renderer.setSeriesShape(1, new Rectangle(6, 6));
        renderer.setSeriesShapesFilled(0, true);
        renderer.setSeriesShapesFilled(1, true);
        // Original points: large opaque circles that stand out on top.
        renderer.setSeriesPaint(2, new Color(10, 60, 160));
        renderer.setSeriesPaint(3, new Color(190, 45, 20));
        renderer.setSeriesShape(2, new Ellipse2D.Double(-4, -4, 8, 8));
        renderer.setSeriesShape(3, new Ellipse2D.Double(-4, -4, 8, 8));
        renderer.setSeriesShapesFilled(2, true);
        renderer.setSeriesShapesFilled(3, true);
        renderer.setDrawSeriesLineAsPath(false);

        XYPlot plot = new XYPlot(dataset, domain, range, renderer);
        plot.setBackgroundPaint(Color.WHITE);
        plot.setDomainGridlinesVisible(false);
        plot.setRangeGridlinesVisible(false);

        JFreeChart chart = new JFreeChart("XOR Decision Boundary",
                JFreeChart.DEFAULT_TITLE_FONT, plot, true);
        chart.setBackgroundPaint(Color.WHITE);
        chart.getLegend().setItemFont(new Font(Font.SANS_SERIF, Font.PLAIN, 13));

        ChartPanel panel = new ChartPanel(chart);
        panel.setPreferredSize(new Dimension(760, 700));

        JFrame frame = new JFrame("XOR Decision Boundary");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(panel);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    private XorDecisionBoundary() {}
}
