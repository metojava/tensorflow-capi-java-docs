package com.example.churn;

import com.example.tfcapi.*;

import java.nio.file.Path;
import java.util.Locale;
import java.util.Random;

/**
 * Project 10 - Telecom customer churn prediction on synthetic data.
 *
 * Shows a mixed feature pipeline: two numeric columns (tenure, monthly
 * charges) are z-score standardized while the categorical contract type is
 * one-hot encoded and the two service flags stay binary. The model is judged
 * with Metrics.printReport and we call out churn recall explicitly - missing
 * a customer who was about to leave is the costly error for the business.
 */
public final class ChurnPrediction {

    // Feature vector layout: [tenureStd, chargesStd,
    //                         contract=month, contract=1yr, contract=2yr,
    //                         techSupport, streaming]  -> 7 inputs.
    private static final int FEATURES = 7;
    private static final String[] NAMES = {"stays", "churns"};
    private static final int CHURN = 1;

    public static void main(String[] args) throws Exception {
        // ---------- 1. Generate synthetic telco customers (seed 21) -------
        Random r = new Random(21);
        int n = 2000;
        float[][] raw = new float[n][5];   // tenure, charges, contract, techSupport, streaming
        int[] y = new int[n];
        int churners = 0;

        for (int i = 0; i < n; i++) {
            int tenure = 1 + r.nextInt(72);                 // months with the company
            float charges = 20 + r.nextFloat() * 100;       // USD per month
            int contract = weightedContract(r);             // 0 month / 1 yr / 2 yr
            int techSupport = r.nextDouble() < 0.30 ? 1 : 0;
            int streaming = r.nextDouble() < 0.60 ? 1 : 0;

            // Ground-truth churn score from the "hidden" generative process.
            double z = 0.04 * (charges - 60) - 0.05 * tenure
                    + (contract == 0 ? 0.9 : 0)
                    - 0.6 * techSupport + 0.3 * streaming - 2.2;
            double p = sigmoid(z) + r.nextGaussian() * 0.02; // light label noise
            p = Math.max(0.001, Math.min(0.999, p));
            int label = r.nextDouble() < p ? 1 : 0;

            raw[i] = new float[]{tenure, charges, contract, techSupport, streaming};
            y[i] = label;
            churners += label;
        }
        System.out.printf("Generated %d customers (%d churners = %.1f%%)%n",
                n, churners, 100.0 * churners / n);

        // ---------- 2. Seeded shuffle + 70/30 split -----------------------
        int[] order = seededOrder(n, 42);
        int nTrain = (int) (n * 0.7);
        int nTest = n - nTrain;
        float[] trainX = new float[nTrain * FEATURES];
        int[] trainY = new int[nTrain];
        float[] testX = new float[nTest * FEATURES];
        int[] testY = new int[nTest];
        for (int i = 0; i < n; i++) {
            if (i < nTrain) copyRow(raw, y, order[i], trainX, trainY, i);
            else copyRow(raw, y, order[i], testX, testY, i - nTrain);
        }

        // ---------- 3. Standardize NUMERIC columns only -------------------
        // One-hot columns are already 0/1 - rescaling them would only distort
        // their meaning, so stats come from columns 0..1 of the training set.
        float[] mean = numericMean(trainX, nTrain);
        float[] std = numericStd(trainX, nTrain, mean);
        standardizeNumerics(trainX, mean, std);
        standardizeNumerics(testX, mean, std);

        // ---------- 4. Build and train the MLP ----------------------------
        // Label smoothing (0.05) keeps the network from pushing probabilities
        // to 0/1 on noisy labels - it visibly reduces overfitting here.
        try (Sequential net = new Sequential(Optimizer.adam(0.01f),
                Sequential.Loss.SPARSE_CROSS_ENTROPY, 0.05f,
                new Input(FEATURES),
                new Dense(16), new ReLU(),
                new Dropout(0.2f),           // regularization: 2000 rows is small
                new Dense(8), new ReLU(),
                new Dense(2))) {

            net.fit(trainX, trainY, 150, 32, testX, testY, true);

            // ---------- 5. Evaluate ---------------------------------------
            float[] probs = net.predict(testX);
            double loss = 0;
            int correct = 0;
            int[][] conf = new int[2][2];
            for (int i = 0; i < nTest; i++) {
                int pred = Metrics.argmax(probs, i * 2, 2);
                conf[testY[i]][pred]++;
                if (pred == testY[i]) correct++;
                loss += -(testY[i] == CHURN
                        ? Math.log(Math.max(probs[i * 2 + 1], 1e-12))
                        : Math.log(Math.max(probs[i * 2], 1e-12)));
            }
            Metrics.printReport(loss / nTest, (double) correct / nTest, conf, NAMES);

            // Highlight the business-critical number: churn recall.
            double tp = conf[CHURN][CHURN], fn = conf[CHURN][0], fp = conf[0][CHURN];
            System.out.printf("Churn recall = %.3f (%d of %d churners caught), precision = %.3f%n",
                    tp / Math.max(tp + fn, 1e-9), (int) tp, (int) (tp + fn),
                    tp / Math.max(tp + fp, 1e-9));

            // Retention teams do not need a 0.5 cut-off - they contact the
            // highest-risk customers. Lower thresholds trade precision for
            // recall; note how recall climbs while precision falls.
            for (double t : new double[]{0.50, 0.35, 0.25}) {
                int[][] c = new int[2][2];
                for (int i = 0; i < nTest; i++) {
                    int pred = probs[i * 2 + CHURN] >= t ? CHURN : 0;
                    c[testY[i]][pred]++;
                }
                double rcl = c[CHURN][CHURN] / Math.max(c[CHURN][CHURN] + c[CHURN][0], 1e-9);
                double prc = c[CHURN][CHURN] / Math.max(c[CHURN][CHURN] + c[0][CHURN], 1e-9);
                System.out.printf(Locale.ROOT,
                        "threshold P(churn)>=%.2f -> churn recall=%.3f precision=%.3f%n", t, rcl, prc);
            }

            // Demo: score three brand-new customer profiles.
            float[][] demo = {
                    {36, 45, 2, 1, 1},   // loyal, cheap 2-year plan with support -> stays?
                    {3, 95, 0, 0, 1},    // brand-new, expensive month-to-month -> churns?
                    {12, 70, 0, 1, 0},   // short tenure month-to-month but has support
            };
            System.out.println("Demo predictions on new customers:");
            for (float[] cust : demo) {
                float[] in = encodeStandardized(cust, mean[0], std[0], mean[1], std[1]);
                float[] p = net.predict(in);
                System.out.printf("  tenure=%.0f charges=%.0f contract=%d support=%d stream=%d"
                                + " -> P(churn)=%.3f (%s)%n",
                        cust[0], cust[1], (int) cust[2], (int) cust[3], (int) cust[4],
                        p[1], p[1] >= 0.5 ? "churns" : "stays");
            }

            net.saveTo(Path.of("target", "churn-model.bin"));
            System.out.println("Model saved to target/churn-model.bin");
        }
    }

    /* ------------------------- encoding helpers ------------------------- */

    /** Builds one raw 7-dim input row (one-hot contract included, no scaling). */
    static float[] encodeRaw(float[] c) {
        return new float[]{
                c[0], c[1],
                c[2] == 0 ? 1 : 0, c[2] == 1 ? 1 : 0, c[2] == 2 ? 1 : 0,
                c[3], c[4]
        };
    }

    /** Same layout but with train-set standardization applied to the numerics. */
    static float[] encodeStandardized(float[] c, float mTenure, float sTenure,
                                      float mChg, float sChg) {
        float[] e = encodeRaw(c);
        e[0] = (e[0] - mTenure) / (sTenure + 1e-8f);
        e[1] = (e[1] - mChg) / (sChg + 1e-8f);
        return e;
    }

    static void copyRow(float[][] raw, int[] y, int row, float[] dst, int[] dstY, int dstRow) {
        // Copy raw numerics first; standardization happens later in bulk.
        System.arraycopy(encodeRaw(raw[row]), 0, dst, dstRow * FEATURES, FEATURES);
        dstY[dstRow] = y[row];
    }

    /* --------------------- math / split helpers ------------------------- */

    static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    /** Month-to-month dominates real telco bases: weights .55/.20/.25. */
    static int weightedContract(Random r) {
        double u = r.nextDouble();
        return u < 0.55 ? 0 : u < 0.75 ? 1 : 2;
    }

    static int[] seededOrder(int n, long seed) {
        int[] o = new int[n];
        for (int i = 0; i < n; i++) o[i] = i;
        Random r = new Random(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1), t = o[i]; o[i] = o[j]; o[j] = t;
        }
        return o;
    }

    static float[] numericMean(float[] d, int rows) {
        float[] m = new float[2];
        for (int i = 0; i < rows; i++) {
            m[0] += d[i * FEATURES];
            m[1] += d[i * FEATURES + 1];
        }
        m[0] /= rows;
        m[1] /= rows;
        return m;
    }

    static float[] numericStd(float[] d, int rows, float[] mean) {
        float[] s = new float[2];
        for (int i = 0; i < rows; i++)
            for (int f = 0; f < 2; f++) {
                float diff = d[i * FEATURES + f] - mean[f];
                s[f] += diff * diff;
            }
        for (int f = 0; f < 2; f++)
            s[f] = (float) Math.sqrt(s[f] / Math.max(rows - 1, 1));
        return s;
    }

    static void standardizeNumerics(float[] d, float[] mean, float[] std) {
        for (int i = 0; i < d.length / FEATURES; i++)
            for (int f = 0; f < 2; f++)
                d[i * FEATURES + f] = (d[i * FEATURES + f] - mean[f]) / (std[f] + 1e-8f);
    }

    private ChurnPrediction() {}
}
