package org.example;

import put.dea.robustness.CCRExtremeEfficiencies;
import put.dea.robustness.ProblemData;

import me.tongfei.progressbar.ProgressBar;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class CsvCandidateEfficiencyExporter {

    private static final List<String> INPUT_NAMES = List.of("i1", "i2", "i3", "i4");
    private static final List<String> OUTPUT_NAMES = List.of("o1", "o2");

    private static final double EFFICIENCY_TOL = 1e-9;

    public static void main(String[] args) throws IOException {
        String inputCsvPath = "output/dea_search_space_RZE.csv";
        String outputCsvPath = "output/dea_search_space_RZE_results.csv";

        List<String> selectedReferenceAirports = List.of("SZZ", "LCJ", "KAT");

        CsvCandidateEfficiencyExporter exporter = new CsvCandidateEfficiencyExporter();
        exporter.run(inputCsvPath, outputCsvPath, selectedReferenceAirports);

        System.out.println("Done. Results saved to: " + outputCsvPath);
    }

    public void run(
            String inputCsvPath,
            String outputCsvPath,
            List<String> selectedReferenceAirports
    ) throws IOException {

        Map<String, AirportRecord> baseAirports = buildBaseAirportMap();
        List<CandidateRow> candidates = readCandidatesFromCsv(inputCsvPath);

        CCRExtremeEfficiencies extremeEfficiencies = new CCRExtremeEfficiencies();

        Path outputPath = Path.of(outputCsvPath);
        if (outputPath.getParent() != null) {
            Files.createDirectories(outputPath.getParent());
        }

        try (BufferedWriter writer = Files.newBufferedWriter(outputPath)) {
            writeHeader(writer, selectedReferenceAirports);
            try (ProgressBar pb = new ProgressBar("DEA candidates", candidates.size())) {
                for (CandidateRow candidate : candidates) {
                    DataBundle bundle = buildProblemDataForCandidate(
                            baseAirports,
                            candidate,
                            selectedReferenceAirports
                    );

                    List<Double> maxEfficiencies = extremeEfficiencies.maxEfficiencyForAll(bundle.data());

                    ResultRow resultRow = buildResultRow(
                            candidate,
                            bundle.dmuNames(),
                            maxEfficiencies,
                            selectedReferenceAirports
                    );

                    writeResultRow(writer, resultRow, selectedReferenceAirports);
                    pb.step();
                }
            }
        }
    }

    private void writeHeader(BufferedWriter writer, List<String> selectedReferenceAirports) throws IOException {
        List<String> cols = new ArrayList<>();

        cols.add("candidate_name");
        cols.add("i1");
        cols.add("i2");
        cols.add("i3");
        cols.add("i4");
        cols.add("o1");
        cols.add("o2");

        cols.add("candidate_efficiency");
        cols.add("candidate_efficient");

        for (String ref : selectedReferenceAirports) {
            cols.add(ref + "_efficiency");
        }

        writer.write(String.join(",", cols));
        writer.newLine();
    }

    private ResultRow buildResultRow(
            CandidateRow candidate,
            List<String> dmuNames,
            List<Double> maxEfficiencies,
            List<String> selectedReferenceAirports
    ) {
        int candidateIdx = dmuNames.indexOf(candidate.name());
        if (candidateIdx < 0) {
            throw new IllegalStateException("Candidate not found in dmuNames: " + candidate.name());
        }

        double candidateEfficiency = maxEfficiencies.get(candidateIdx);
        boolean candidateEfficient = isEfficient(candidateEfficiency);

        Map<String, Double> referenceEfficiencies = new LinkedHashMap<>();
        for (String ref : selectedReferenceAirports) {
            int refIdx = dmuNames.indexOf(ref);
            if (refIdx < 0) {
                throw new IllegalStateException("Reference airport not found in dmuNames: " + ref);
            }
            referenceEfficiencies.put(ref, maxEfficiencies.get(refIdx));
        }

        return new ResultRow(
                candidate.name(),
                candidate.i1(),
                candidate.i2(),
                candidate.i3(),
                candidate.i4(),
                candidate.o1(),
                candidate.o2(),
                candidateEfficiency,
                candidateEfficient,
                referenceEfficiencies
        );
    }

    private void writeResultRow(
            BufferedWriter writer,
            ResultRow row,
            List<String> selectedReferenceAirports
    ) throws IOException {
        List<String> cols = new ArrayList<>();

        cols.add(escapeCsv(row.candidateName()));
        cols.add(formatDouble(row.i1()));
        cols.add(formatDouble(row.i2()));
        cols.add(formatDouble(row.i3()));
        cols.add(formatDouble(row.i4()));
        cols.add(formatDouble(row.o1()));
        cols.add(formatDouble(row.o2()));

        cols.add(formatDouble(row.candidateEfficiency()));
        cols.add(Boolean.toString(row.candidateEfficient()));

        for (String ref : selectedReferenceAirports) {
            Double eff = row.referenceEfficiencies().get(ref);
            cols.add(formatDouble(eff));
        }

        writer.write(String.join(",", cols));
        writer.newLine();
    }

    private boolean isEfficient(double efficiency) {
        return Math.abs(efficiency - 1.0) <= EFFICIENCY_TOL || efficiency > 1.0;
    }

    private String formatDouble(double value) {
        return String.format(Locale.US, "%.10f", value);
    }

    private String escapeCsv(String value) {
        if (value.contains(",") || value.contains("\"") || value.contains("\n")) {
            return "\"" + value.replace("\"", "\"\"") + "\"";
        }
        return value;
    }

    private DataBundle buildProblemDataForCandidate(
            Map<String, AirportRecord> baseAirports,
            CandidateRow candidate,
            List<String> selectedReferenceAirports
    ) {
        List<String> dmuNames = new ArrayList<>();
        List<double[]> inputsList = new ArrayList<>();
        List<double[]> outputsList = new ArrayList<>();

        for (String airportName : selectedReferenceAirports) {
            AirportRecord airport = baseAirports.get(airportName);
            if (airport == null) {
                throw new IllegalArgumentException("Unknown reference airport: " + airportName);
            }

            dmuNames.add(airport.name());
            inputsList.add(Arrays.copyOf(airport.inputs(), airport.inputs().length));
            outputsList.add(Arrays.copyOf(airport.outputs(), airport.outputs().length));
        }

        dmuNames.add(candidate.name());
        inputsList.add(new double[]{candidate.i1(), candidate.i2(), candidate.i3(), candidate.i4()});
        outputsList.add(new double[]{candidate.o1(), candidate.o2()});

        double[][] inputs = inputsList.toArray(new double[0][]);
        double[][] outputs = outputsList.toArray(new double[0][]);

        ProblemData data = new ProblemData(inputs, outputs, INPUT_NAMES, OUTPUT_NAMES);

        return new DataBundle(data, dmuNames);
    }

    private Map<String, AirportRecord> buildBaseAirportMap() {
        List<String> alternativeNames = List.of(
                "WAW", "KRK", "KAT", "WRO", "POZ", "LCJ", "GDN", "SZZ", "BZG", "RZE", "IEG"
        );

        double[][] inputs = new double[][]{
                {10.5, 36, 129.4, 7},
                {3.1, 19, 31.6, 7.9},
                {3.6, 32, 57.4, 10.5},
                {1.5, 12, 18, 3},
                {1.5, 10, 24, 4},
                {0.6, 12, 24, 3.9},
                {1.0, 15, 42.9, 2.5},
                {0.7, 10, 25.7, 1.9},
                {0.3, 6, 3.4, 1.2},
                {0.6, 6, 11.3, 2.7},
                {0.1, 10, 63.4, 3}
        };

        double[][] outputs = new double[][]{
                {9.5, 129.7},
                {2.9, 31.3},
                {2.4, 21.1},
                {1.5, 18.8},
                {1.3, 16.2},
                {0.3, 4.2},
                {2.0, 23.6},
                {0.3, 4.2},
                {0.3, 6.2},
                {0.3, 3.5},
                {0.005, 0.61}
        };

        Map<String, AirportRecord> result = new LinkedHashMap<>();
        for (int i = 0; i < alternativeNames.size(); i++) {
            result.put(
                    alternativeNames.get(i),
                    new AirportRecord(
                            alternativeNames.get(i),
                            Arrays.copyOf(inputs[i], inputs[i].length),
                            Arrays.copyOf(outputs[i], outputs[i].length)
                    )
            );
        }

        return result;
    }

    private List<CandidateRow> readCandidatesFromCsv(String csvPath) throws IOException {
        List<CandidateRow> rows = new ArrayList<>();

        try (BufferedReader reader = Files.newBufferedReader(Path.of(csvPath))) {
            String headerLine = reader.readLine();
            if (headerLine == null) {
                throw new IllegalArgumentException("Empty CSV: " + csvPath);
            }

            String[] headers = headerLine.split(",", -1);
            Map<String, Integer> idx = new HashMap<>();
            for (int i = 0; i < headers.length; i++) {
                idx.put(headers[i].trim(), i);
            }

            require(idx, "name");
            require(idx, "i1");
            require(idx, "i2");
            require(idx, "i3");
            require(idx, "i4");
            require(idx, "o1");
            require(idx, "o2");

            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }

                String[] parts = line.split(",", -1);

                rows.add(new CandidateRow(
                        parts[idx.get("name")].trim(),
                        Double.parseDouble(parts[idx.get("i1")].trim()),
                        Double.parseDouble(parts[idx.get("i2")].trim()),
                        Double.parseDouble(parts[idx.get("i3")].trim()),
                        Double.parseDouble(parts[idx.get("i4")].trim()),
                        Double.parseDouble(parts[idx.get("o1")].trim()),
                        Double.parseDouble(parts[idx.get("o2")].trim())
                ));
            }
        }

        return rows;
    }

    private void require(Map<String, Integer> idx, String col) {
        if (!idx.containsKey(col)) {
            throw new IllegalArgumentException("Missing column in CSV: " + col);
        }
    }

    private record CandidateRow(
            String name,
            double i1,
            double i2,
            double i3,
            double i4,
            double o1,
            double o2
    ) {}

    private record AirportRecord(
            String name,
            double[] inputs,
            double[] outputs
    ) {}

    private record DataBundle(
            ProblemData data,
            List<String> dmuNames
    ) {}

    private record ResultRow(
            String candidateName,
            double i1,
            double i2,
            double i3,
            double i4,
            double o1,
            double o2,
            double candidateEfficiency,
            boolean candidateEfficient,
            Map<String, Double> referenceEfficiencies
    ) {}
}