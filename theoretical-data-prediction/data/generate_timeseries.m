function generate_fig3_timeseries
    % ------------------ Parameters ------------------
    params.VS   = 5;
    params.R    = 1000;
    params.L1   = 15e-6;
    params.L2   = 220e-6;
    params.C1   = 220e-12;
    params.C2   = 5e-12;
    params.Vth  = 0.6;
    params.beta = 200;

    % Initial conditions at t = 0
    uC1_0 = params.Vth;
    uC2_0 = params.Vth;
    iL1_0 = params.Vth / params.R;
    iL2_0 = params.Vth / params.R;
    y0 = [uC1_0; uC2_0; iL1_0; iL2_0];

    % Time span: 0 to 320 us, dt = 32e-6 / 10000
    t0   = 0;
    tmax = 10e-5;
    dt   = 32e-6 / 10000;
    tspan = t0:dt:tmax;

    opts = odeset('RelTol',1e-6);

    [t, y] = ode45(@(t,y) circuitODE(t,y,params), tspan, y0, opts);

    % Extract variables
    uC1 = y(:,1);
    uC2 = y(:,2);
    iL1 = y(:,3);
    iL2 = y(:,4);

    % Plot: 4 subplots stacked vertically, top to bottom: uC1, uC2, iL1, iL2
    varNames = {'u_{C1}', 'u_{C2}', 'i_{L1}', 'i_{L2}'};
    varData  = {uC1, uC2, iL1, iL2};

    figure;
    for k = 1:4
        subplot(4,1,k);
        plot(t*1e6, varData{k}, 'LineWidth', 1);
        ylabel(varNames{k});
        grid on;
        axis tight;
        if k < 4
            set(gca, 'XTickLabel', []);
        else
            xlabel('Time [\mus]');
        end
        if k == 1
            title('Time series of circuit variables (0-32 \mus)');
        end
    end

    saveas(gcf, 'timeseries_32us.png');
    disp('Saved: timeseries_32us.png');

    % Save CSV: 4 rows x 100000 columns, no header, comma-separated scientific notation
    % Row 1: uC1, Row 2: uC2, Row 3: iL1, Row 4: iL2
    csvFile = 'output_320.csv';
    fid = fopen(csvFile, 'w');
    allData = {uC1', uC2', iL1', iL2'};
    for row = 1:4
        data = allData{row};
        for col = 1:length(data)
            if col < length(data)
                fprintf(fid, '%.18e,', data(col));
            else
                fprintf(fid, '%.18e', data(col));
            end
        end
        if row < 4
            fprintf(fid, '\n');
        end
    end
    fclose(fid);
    disp(['Saved: ', csvFile]);
    disp(['Points: ', num2str(length(t))]);
end

function dydt = circuitODE(~, y, p)
    uC1 = y(1);
    uC2 = y(2);
    iL1 = y(3);
    iL2 = y(4);

    Gamma = max(iL1,0);
    iT = p.beta * Gamma * tanh(uC2/(2*p.Vth));

    duC1 = (p.VS - uC1)/(p.R*p.C1) - (iL1 + iL2)/p.C1;
    duC2 = (iL2 - iT)/p.C2;
    diL1 = (uC1 - p.Vth)/p.L1;
    diL2 = (uC1 - uC2)/p.L2;

    dydt = [duC1; duC2; diL1; diL2];
end
