///////////////////////////////////////////////////////////////////////////////
//

let cards = [];

let variables = {
  'K':       {repr: 'K',                                                         meaning: 'Capital',                                   thread: 'capital'},

  'K_G':     {meaning: 'Capital dedicated to G&S',                               thread: 'goods.capital'},
  'Cog_G':   {meaning: 'Cognitive input to G&S',                                 thread: 'goods.cognitive_output'},
  'T_{G,i}': {meaning: 'Output of the \\(i\\)-th G&S task',                      thread: 'goods.task_input'},                          // <-- TODO
  'C_{G,0}': {meaning: 'Output of the non-AI G&S compute tasks',                 thread: 'goods.compute_task_input[0]'},               // <-- TODO
  'C_{G,i}': {meaning: 'Compute allocated to the \\(i\\)-th G&S task',           thread: 'goods.compute_task_input'},                  // <-- TODO
  'L_{G,i}': {meaning: 'Labour allocated to the \\(i\\)-th G&S task',            thread: 'goods.labour_task_input'},                   // <-- TODO

  'R_H':     {meaning: 'Research input to hardware R&D',                         thread: 'hardware_performance.rnd_input'},
  'R_S':     {meaning: 'Research input to software R&D',                         thread: 'software.rnd_input'},

  'Cog_H':   {meaning: 'Cognitive input to hardware R&D',                        thread: 'hardware_rnd.cognitive_output'},
  'T_{H,i}': {meaning: 'Output of the \\(i\\)-th hardware R&D task',             thread: 'hardware_rnd.task_input'},                   // <-- TODO
  'C_{H,0}': {meaning: 'Output of the non-AI hardware R&D compute tasks',        thread: 'hardware_rnd.compute_task_input[0]'},        // <-- TODO
  'C_{H,i}': {meaning: 'Compute allocated to the \\(i\\)-th hardware R&D task',  thread: 'hardware_rnd.compute_task_input'},           // <-- TODO
  'L_{H,i}': {meaning: 'Labour allocated to the \\(i\\)-th hardware R&D task',   thread: 'hardware_rnd.labour_task_input'},            // <-- TODO

  'Cog_S':   {meaning: 'Cognitive input to software R&D',                        thread: 'software_rnd.cognitive_output'},
  'T_{S,i}': {meaning: 'Output of the \\(i\\)-th software R&D task',             thread: 'software_rnd.task_input'},                   // <-- TODO
  'C_{S,0}': {meaning: 'Output of the non-AI software R&D compute tasks',        thread: 'software_rnd.compute_task_input[0]'},        // <-- TODO
  'C_{S,i}': {meaning: 'Compute allocated to the \\(i\\)-th software R&D task',  thread: 'software_rnd.compute_task_input'},           // <-- TODO
  'L_{S,i}': {meaning: 'Labour allocated to the \\(i\\)-th software R&D task',   thread: 'software_rnd.labour_task_input'},            // <-- TODO

  'GWP':     {meaning: 'Gross world product',                                    thread: 'gwp'},

  'L':       {meaning: 'Labour',                                                 thread: 'labour'},
  'TFP':     {meaning: 'Total factor productivity',                              thread: 'goods.tfp'},

  'HS':      {meaning: 'Hardware stock',                                         thread: 'hardware'},

  'H':       {meaning: 'Hardware efficiency',                                    thread: 'hardware_performance.v'},

  'C':       {meaning: 'Compute',                                                thread: 'compute'},
  'S':       {meaning: 'Software efficiency level',                              thread: 'software.v'},

  'F_C':     {meaning: 'Fraction of GWP used to purchase new hardware',          thread: 'compute_investment'},
  'F_{C,G}': {meaning: 'Fraction of effective compute assigned to G&S',          thread: 'frac_compute.goods.v'},
  'F_{C,H}': {meaning: 'Fraction of effective compute assigned to hardware R&D', thread: 'frac_compute.hardware_rnd.v'},
  'F_{C,S}': {meaning: 'Fraction of effective compute assigned to software R&D', thread: 'frac_compute.software_rnd.v'},
  'F_{C,T}': {meaning: 'Fraction of effective compute assigned to training',     thread: 'frac_compute.training.v'},
  'F_{K,G}': {meaning: 'Fraction of capital assigned to G&S',                    thread: 'frac_capital.goods.v'},
  'F_{K,H}': {meaning: 'Fraction of capital assigned to hardware R&D',           thread: 'frac_capital.hardware_rnd.v'},
  'F_{L,G}': {meaning: 'Fraction of labor assigned to G&S',                      thread: 'frac_labour.goods.v'},
  'F_{L,H}': {meaning: 'Fraction of labor assigned to hardware R&D',             thread: 'frac_labour.hardware_rnd.v'},
  'F_{L,S}': {meaning: 'Fraction of labor assigned to software R&D',             thread: 'frac_labour.software_rnd.v'},

  'C_T':     {meaning: 'Largest training run',                                   thread: 'biggest_training_run'},

  'P_H':     {meaning: 'Penalty factor for hardware R&D',                        thread: 'hardware_rnd.ceiling_penalty'},
  'Q_H':     {meaning: 'Adjusted cumulative inputs to hardware R&D',             thread: 'hardware_performance.cumulative_rnd_input'},

  'P_S':     {meaning: 'Penalty factor for software R&D',                        thread: 'software_rnd.ceiling_penalty'},
  'Q_S':     {meaning: 'Adjusted cumulative inputs to software R&D',             thread: 'software.cumulative_rnd_input'},

  'A_G':     {meaning: 'Automation index for G&S',                               thread: 'goods.at',                                   yscale: 'linear'},
  'A_R':     {meaning: 'Automation index for R&D',                               thread: 'rnd.at',                                     yscale: 'linear'},
}

let parameters = {
  'g_L':          {meaning: 'Growth rate of labour',                                                                                     notes: ''},
  'g_{TFP}':      {meaning: 'Growth rate of TFP',                                                                                        notes: ''},
  'h_d':          {meaning: 'Hardware delay',                                                                                            notes: ''},
  'd_C':          {meaning: 'Hardware depreciation rate',                                                                                notes: ''},
  '\\tau_{G,i}':  {meaning: 'Training requirements for automation of the \\(i\\)-th G&S task',                                           notes: ''},
  '\\tau_{R,i}':  {meaning: 'Training requirements for automation of the \\(i\\)-th R&D task',                                           notes: ''},

  's_K':          {meaning: 'Savings rate',                                                                                              notes: ''},

  '\\alpha_G':    {meaning: 'Weight associated to capital in the G&S CES production function',                                           notes: ''},
  '\\rho_G':      {meaning: 'CES substitution for G&S',                                                                                  notes: ''},
  '\\beta_{G,i}': {meaning: 'Task weights in the G&S cognitive CES function',                                                            notes: ''},
  '\\psi_G':      {meaning: 'Substitution parameter in the G&S cognitive CES function',                                                  notes: ''},
  '\\eta_{G,i}':  {meaning: 'Compute-to-labour efficiency for the \\(i\\)-th G&S task',                                                 notes: ''},

  '\\alpha_R':    {meaning: 'Weight associated to capital in the hardware R&D CES production function',                                  notes: ''},

  '\\rho_R':      {meaning: 'CES substitution for hardware and software R&D',                                                            notes: ''},
  '\\beta_{R,i}': {meaning: 'Task weights in the hardware and software R&D cognitive CES function',                                      notes: ''},
  '\\psi_R':      {meaning: 'Substitution parameter in the hardware and software R&D cognitive CES function',                            notes: ''},
  '\\eta_{R,i}':  {meaning: 'Compute-to-labour efficiency for the \\(i\\)-th hardware and software R&D task',                           notes: ''},

  '\\alpha_S':    {meaning: 'Weight associated to physical compute dedicated to experiments in the softwre R&D CES production function', notes: ''},
  '\\rho_S':      {meaning: 'CES substitution for software R&D',                                                                         notes: ''},
  '\\zeta':       {meaning: 'Discounting factor for physical compute dedicated to experiments',                                          notes: ''},

  'r_H':          {meaning: 'Efficiency of the returns to hardware R&D',                                                                 notes: ''},
  'r_S':          {meaning: 'Efficiency of the returns to software R&D',                                                                 notes: ''},

  'U_C':          {meaning: 'Maximum fraction of GWP used to purchase new hardware',                                                     notes: ''},
  'U_{C,G}':      {meaning: 'Maximum fraction of effective compute assigned to G&S',                                                     notes: ''},
  'U_{C,H}':      {meaning: 'Maximum fraction of effective compute assigned to hardware R&D',                                            notes: ''},
  'U_{C,S}':      {meaning: 'Maximum fraction of effective compute assigned to software R&D',                                            notes: ''},
  'U_{C,T}':      {meaning: 'Maximum fraction of effective compute assigned to training',                                                notes: ''},
  'U_{K,G}':      {meaning: 'Maximum fraction of capital assigned to G&S',                                                               notes: ''},
  'U_{K,H}':      {meaning: 'Maximum fraction of capital assigned to hardware R&D',                                                      notes: ''},
  'U_{L,G}':      {meaning: 'Maximum fraction of labor assigned to G&S',                                                                 notes: ''},
  'U_{L,H}':      {meaning: 'Maximum fraction of labor assigned to hardware R&D',                                                        notes: ''},
  'U_{L,S}':      {meaning: 'Maximum fraction of labor assigned to software R&D',                                                        notes: ''},

  'U_H':          {meaning: 'Maximum hardware efficiency',                                                                               notes: ''},
  'U_S':          {meaning: 'Maximum software efficiency',                                                                               notes: ''},

  'H_0':          {meaning: 'Initial hardware efficiency',                                                                               notes: ''},
  'S_0':          {meaning: 'Initial software efficiency',                                                                               notes: ''},

  'G_C':          {meaning: 'Growth of the fraction of GWP used to purchase new hardware',                                               notes: ''},
  'G_{C,G}':      {meaning: 'Growth of the fraction of effective compute assigned to G&S',                                               notes: ''},
  'G_{C,H}':      {meaning: 'Growth of the fraction of effective compute assigned to hardware R&D',                                      notes: ''},
  'G_{C,S}':      {meaning: 'Growth of the fraction of effective compute assigned to software R&D',                                      notes: ''},
  'G_{C,T}':      {meaning: 'Growth of the fraction of effective compute assigned to training',                                          notes: ''},
  'G_{K,G}':      {meaning: 'Growth of the fraction of capital assigned to G&S',                                                         notes: ''},
  'G_{K,H}':      {meaning: 'Growth of the fraction of capital assigned to hardware R&D',                                                notes: ''},
  'G_{L,G}':      {meaning: 'Growth of the fraction of labor assigned to G&S',                                                           notes: ''},
  'G_{L,H}':      {meaning: 'Growth of the fraction of labor assigned to hardware R&D',                                                  notes: ''},
  'G_{L,S}':      {meaning: 'Growth of the fraction of labor assigned to software R&D',                                                  notes: ''},

  '\\lambda':     {meaning: 'R&D parallelization penalty',                                                                               notes:  ''},
}

for (let name in variables) {
  variables[name].name = name;
}

for (let name in parameters) {
  parameters[name].name = name;
}

let automationCard = {
  id: 'automation',

  title: 'Automation',

  abstract: `The automation module takes as input effective compute and determines to which degree and how efficiently we can substitute labour with compute in the production functions.`,

  explanation: `
    <p>The automation indices \\(A_G\\) and \\(A_R\\) determine the number of cognitive tasks that can be automated in G&S and R&D production.</p>

    <p>The amount of effective compute dedicated to the largest training run to date \\(C_T\\) is a fraction \\(F_{C,T}\\) of the total available compute:</p>

    <div>\\[C_T = F_{C,T} C\\]</div>

    <p id="automation-index">To determine the automation index for goods and services, we compare \\(C_T\\) to the automation training requirements \\(\\tau_{G,1}, ..., \\tau_{G,N}\\). We set the automation index \\(A_G\\) to the largest task index \\(i\\) such that \\(\\tau_{G,i} < C_T\\).</p>

    <div>\\[A_G = \\max_i \\{i : \\tau_{G,i} < C_T\\} \\]</div>

    <p>The distribution of the training requirements is parameterized in terms of the effective FLOP needed for full automation of cognitive tasks \\(\\tau_{G,N}\\) and the ratio between that and the requirements for 20% automation, which we call the FLOP gap.</p>

    <p>For example, for the best-guess values of the effective FLOP gap and the full automation requirements \\(\\tau_{G,N}\\) the spread of \\(\\tau_{G,i}\\) looks like this:</p>

    <figure>
        <img src="img/training-reqs.png">
        <figcaption>Distribution of training requirements \\(\\tau_G,i\\)</figcaption>
    </figure>

    <p>These scale requirements were chosen with two criteria in mind:</p>

    <ol>
      <li>Each additional OOM of training unlocks more tasks than the last.</li>
      <li>The effective FLOP gap from ~1% to 20% is half the FLOP gap from 20% to 100%.</li>
    </ol>

    <p>You can read more about these design choices in <a href="https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.epvo05wz8jt1">this report section</a>.</p>

    <div class="learn-more">
      <p>The largest automation training threshold \\(\\tau_{G,N}\\) corresponding to the training cost of AGI is a parameter of the model. The distribution of the rest of the thresholds of the other tasks is derived using the following rules:</p>

      <ol>
        <li>
          The training cost \\(\\tau_{\\lfloor{20\\%N}\\rfloor}\\) corresponding to 20% automation of G&S is set such that \\(\\tau_{G,N} / \\tau_{G, \\lfloor{20\\%N}\\rfloor}\\) is equal to the effective FLOP gap.
        </li>

        <li>
          The following training costs are set by hand:
          <p>
          \\(\\tau_{G, \\lfloor{50\\%N}\\rfloor} := \\tau_{G,N}/(\\text{FLOP gap})^{4/7}\\),
          \\(\\tau_{G, \\lfloor{20\\%N}\\rfloor} :=  \\tau_{G,N}/(\\text{FLOP gap})\\),
          \\(\\tau_{G, \\lfloor{10\\%N}\\rfloor} := \\tau_{G,N}/(\\text{FLOP gap})^{8.5/7}\\),
          \\(\\tau_{G, \\lfloor{5\\%N}\\rfloor} := \\tau_{G,N}/(\\text{FLOP gap})^{9.5/7}\\),
          \\(\\tau_{G, 0} := \\tau_{G,N}/(\\text{FLOP gap})^{10.5/7}\\)
          </p>
        </li>

        <li>
          The remaining task thresholds \\(\\tau_{G,i}\\) are geometrically interpolated from the closest ones, so that, e.g. all \\(\\tau_{G,i}\\) for \\(\\lfloor{20\\%N}\\rfloor \\le i \\le \\lfloor{50\\%N}\\rfloor\\) are equidistant in log-space.
        </li>
      </ol>
    </div>

    <p>The R&D automation index \\(A_R\\) is set identically, substituting \\(T_{G,N}\\) for \\(T_{R,N} := T_{G,N} / \\text{ratio}\\), where \\(\\text{ratio}\\) is a pre-specified ratio of training requirements between goods and services and R&D tasks.</p>

    <div>\\[A_R = \\max_i \\{i : \\tau_{R,i} < C_T\\} \\]</div>

    <p>Finally, we follow an identical procedure as for the training requirements to determine the runtime requirements for each task. We then define the compute-to-labour ratio of the cognitive tasks \\(\\eta_{x,i}\\) as the inverse of the runtime requirements. The result is that tasks that are easier to automate also require less effective compute to replace labour.</p>

    <br>

    <p>In one modification of the model, we allow the automation of tasks where we haven't reached the corresponding effective compute automation threshold \\(\\tau_{G,i}\\) at the cost of a lower compute-to-labour efficiency \\(\\eta_{G,i}\\). This essentially allows AI developers to 'tradeoff' training and runtime effective compute -- use excess runtime compute to make up for a smaller training run.</p>

    <div class="learn-more">
      <p>As long as the training compute \\(C_T\\) is within a threshold \\(\\tau_{G,i} / C_T < \\text{Max tradeoff}\\), the task is considered automated, with a compute-to-labour efficiency of:</p>

      <div>\\[\\eta_{G,i} = \\frac{1}{run_{G,i} \\cdot\\left(\\frac{\\tau_{G,i}}{C_T} \\right)^\\delta}\\]</div>

      <p>where \\(run_{G,i}\\) are the base runtime requirements (see section on [automation]) and \\(\\delta > 1\\) is a parameter governing the efficiency of the tradeoff.</p>

      <p>Note that the efficiency increases smoothly beyond the base runtime requirements; runtime requirements never stop getting lower over time as more compute is invested in training larger AI systems.</p>
    </div>
  `,

  equations: [
    'C_T = F_{C,T} C',
    'A_G = \\max_i \\{i : \\tau_{G,i} < C_T\\}',
    'A_R = \\max_i \\{i : \\tau_{R,i} < C_T\\}',
  ],

  variables: [
    'C_T',
    'F_{C,T}',
    'C',
    'A_G',
    'A_R',
  ],

  parameters: [
    '\\tau_{G,i}',
    '\\tau_{R,i}',
  ],
};

cards.push(automationCard);

let rndCard = {
  id: 'rnd',

  title: 'Converting R&D input into improvements to hardware and software',

  abstract: 'The R&D module takes the output of the hardware and software R&D production functions and determines how the hardware efficiency level (measured in physical FLOP/s/$) and the software efficiency level (measured in effective FLOP/physical FLOP) improve in each timestep.',

  explanation: `
    <p>The research inputs \\(R_H\\) and \\(R_S\\) need to be turned into actual performance gains in hardware and software. These quantities are measured in FLOP/year per $ (what we will call the hardware efficiency level \\(H\\)) and effective FLOP per physical FLOP (what we will call the software efficiency level \\(S\\)).</p>

    <p>For this, we use an equation of gradual improvement of hardware and software efficiency derived from <a href="https://www.jstor.org/stable/2138581">(Jones, 1995)</a>:</p>

    <div>\\[\\frac{\\dot H}{H} = r_H \\cdot \\frac{\\dot Q_H}{Q_H}\\]</div>

    <p>where \\(r_H\\) is the efficiency of the returns to hardware and \\(Q_H = \\int^t_{-\\infty} R_H^\\lambda\\) is the cumulative research so far, adjusted by a parameter \\(0 < \\lambda < 1\\) which controls a stepping-on-toes dynamic.</p>

    <div class="learn-more">
      <p>The equation from <a href="https://www.jstor.org/stable/2138581">(Jones, 1995)</a> has the form \\(\\frac{\\dot H}{H} = \\delta R_H^\\lambda H^{-1/r}\\)</p>

      <p>Rearranging and integrating, we get \\(H = (\\delta/r)^{r} (\\int R_H^{\\lambda})^{r} = (\\delta/r)^{r} Q^{r}\\) deriving and diving by \\(H\\) we arrive at the equation above.</p>

      <p>In the code, this is implemented as \\(\\frac{H_{t+\\Delta}}{H_t} = \\frac{(\\delta/r)^{r} Q_{t+\\Delta}^{r}}{(\\delta/r)^{r} Q_{t}^{r}} \\approx \\left(\\frac{Q_{t} + \\Delta R_{H,t}^\\lambda}{Q_{t}}\\right)^r\\)</p>
    </div>

    <p>To account for a ceiling on the physically possible performance, we update the dynamics by multiplying the returns \\(r_H\\) by a penalty factor \\(P_H := \\frac{\\log (U_H/H)}{\\log (U_H/H_0)}\\), where \\(U_H\\) is the upper bound on performance and \\(H_0\\) is the initial performance. When \\(H = U_H\\) the penalty factor equals 0, and no more progress is possible. Initially, the penalty factor equals 1. Each time H doubles, the penalty factor reduces by a constant amount.</p>

    <p>The penalty is included in the dynamic as a factor multiplying the returns:</p>

    <div>\\[\\frac{\\dot H}{H} = {r_H \\cdot P_H} \\cdot\\frac{\\dot Q_H}{Q_H}\\]</div>

    <p>The dynamics for software efficiency are the same as for hardware. We use the same parameter for the parallel penalty \\(\\lambda\\) as in hardware, and software-specific parameters for the returns \\(r_S\\) and the software ceiling \\(U_S\\).</p>

    <div>\\[\\frac{\\dot S}{S} = {r_S \\cdot P_S} \\cdot\\frac{\\dot Q_S}{Q_S}\\]</div>
  `,

  equations: [
    '\\frac{\\dot H}{H} = {r_H \\cdot P_H} \\cdot\\frac{\\dot Q_H}{Q_H}',
    '\\frac{\\dot S}{S} = {r_S \\cdot P_S} \\cdot\\frac{\\dot Q_S}{Q_S}',
    'P_H = \\frac{\\log (U_H/H)}{\\log (U_H/H_0)}',
    'P_S = \\frac{\\log (U_S/S)}{\\log (U_S/S_0)}',
  ],

  variables: [
    'H',
    'P_H',
    'Q_H',

    'S',
    'P_S',
    'Q_S',
  ],

  parameters: [
    'r_H',
    'r_S',

    'U_H',
    'U_S',

    'H_0',
    'S_0',

    '\\lambda',
  ],
};

cards.push(rndCard);

let reinvestmentCard = {
  id: 'reinvestment',

  boxes: ['investment-box', 'reinvestment-box'],

  title: 'Investment / Reinvestment',

  abstract: "The reinvestment module updates the available capital, labour, effective compute and total factor productivity (TFP) based on the model's output. Then it updates how these factors will be split between automation, G&S production and R&D in the next timestep.",

  explanation: `
    <h4>Updating the factors of production</h4>

    <p>Capital varies as usual with the GWP</p>

    <div>\\[\\dot K = s_K \\cdot \\text{GWP}\\]</div>

    <p>where \\(s_K\\) is the savings rate<sup data-tooltip="We considered adding a capital depreciation factor \\(d_K\\) so that capital depreciates over time as \\(\\dot K = s_K GWP - d K\\). However, this didn't significantly affect the model's results, so we removed it." class="footnote" role="doc-noteref">1</sup>.</p>

    <p>Labour and TFP change exogenously at fixed growth rates \\(g_L\\) and \\(g_{TFP}\\):</p>

    <div>\\[\\dot L = g_L L\\]</div>

    <div>\\[\\dot{\\text{TFP}} = g_{\\text{TFP}} \\text{TFP}\\]</div>

    <p>Compute changes in a more complex way. Each timestep, a fraction \\(F_C\\) of the GWP is invested in buying new hardware. This hardware is added to the available hardware stock \\(HS\\), measured in physical FLOP/year.</p>

    <div>\\[\\dot{HS} = F_C \\cdot \\text{GWP} \\cdot H_{t - h_d} - d_C \\cdot HS\\]</div>

    <p>where \\(F_C\\) is the compute investment rate, \\(d_C\\) is the hardware depreciation rate and \\(H_{t - h_d}\\) are the state-of-the-art FLOP/year/$ before a time delay \\(h_d\\). We introduce this delay to signify the time between the design of better hardware and its use in production.</p>

    <p>Finally, the amount of effective compute available is computed as the hardware stock \\(HS\\) times the software level \\(S\\).</p>

    <div>\\[C = HS \\cdot S\\]</div>

    <h4>Allocating resources and wake-up</h4>

    <p>In each timestep, we change how the available production factors are allocated between buying new hardware, the G&S production function, the hardware R&D production function, the software R&D production function and the training of new AI models.</p>

    <table>
      <thead>
        <tr>
          <th>Variable</th>
          <th>Meaning</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>\\(F_C\\)</td>
          <td>Fraction of GWP used to purchase new hardware</td>
        </tr>

        <tr>
          <td>\\(F_{C,G}\\)</td>
          <td>Fraction of effective compute assigned to G&S</td>
        </tr>

        <tr>
          <td>\\(F_{C,H}\\)</td>
          <td>Fraction of effective compute assigned to hardware R&D</td>
        </tr>

        <tr>
          <td>\\(F_{C,S}\\)</td>
          <td>Fraction of effective compute assigned to software R&D</td>
        </tr>

        <tr>
          <td>\\(F_{C,T}\\)</td>
          <td>Fraction of effective compute assigned to training</td>
        </tr>

        <tr>
          <td>\\(F_{K,G}\\)</td>
          <td>Fraction of capital assigned to G&S</td>
        </tr>

        <tr>
          <td>\\(F_{K,H}\\)</td>
          <td>Fraction of capital assigned to hardware R&D</td>
        </tr>

        <tr>
          <td>\\(F_{L,G}\\)</td>
          <td>Fraction of labor assigned to G&S</td>
        </tr>

        <tr>
          <td>\\(F_{L,H}\\)</td>
          <td>Fraction of labor assigned to hardware R&D</td>
        </tr>

        <tr>
          <td>\\(F_{L,S}\\)</td>
          <td>Fraction of labor assigned to software R&D</td>
        </tr>
      </tbody>
    </table>

    <p>These fractions are used to determine the available capital, compute and labour available for each production function. So, for example, the compute available for hardware R&D is equal to \\(C_H = C \\\cdot F_{C,H}\\).</p>

    <p>The fractions are not static, and each grows following the formula:</p>

    <div>\\[\\dot F_X = G_X \\cdot F_X\\]</div>

    <p>where \\(G_X\\) is the growth rate of each fraction. Each fraction is also capped at an upper bound \\(U_X\\).</p>

    <p>The growth rate \\(G_X\\) takes a value \\(g_X\\) in the first period of the simulation, in line with historical data. When the fraction \\(I_G\\) of cognitive tasks for G&S automated exceeds a threshold, the growth rate increases to a "wake-up" value \\(g'_X\\), representing the world "waking-up" to the potential of AI and upping their investment in it.</p>

    <div class="learn-more">
      <p>There are two exceptions.</p>

      <p>Firstly, the above equations do not govern the fractions of each input assigned to G&S. Instead, they are chosen so that the total fraction of each input (across G&S, hardware R&D and software R&D) adds up to 100%:</p>

      <div>\\[S_{C,G} = 1 - S_{C,H} - S_{C,S} - S_{C,T}\\]</div>

      <div>\\[S_{K,G} = 1 - S_{K,H}\\]</div>

      <div>\\[S_{L,G} = 1 - S_{L,H} - S_{L,S}\\]</div>

      <p>as a result, these fractions fall over time (while the others are growing).</p>

      <p>Secondly, the growth rate \\(G_C\\) of the fraction of GWP on compute \\(F_C\\) changes to 0 after the money invested in training AI exceeds a threshold (by default $4B). Then after wake-up it changes from 0 to a new (higher) value. This is a proxy for people's unwillingness to spend very large amounts on a single training run before AI can create large economic value.</p>

      <p>The amount invested in training is assumed to be proportional to 'physical compute allocated to training' /  FLOP/year/$; more specifically we calculate it as \\(C_T / (S \\cdot H_{t - h_d})\\), where \\(C_T = C \\cdot S_{C,T}\\) is the amount of effective compute allocated to training, \\(S\\) is the amount of effective FLOP per physical FLOP  and \\(H_{t - h_d}\\) is the physical FLOP/year/$ that is commercially available.</p>
       
      <p>So the variables for which the exceptions do not apply are the fractions of each input assigned to software R&D and hardware R&D, and the fraction of GDP spent on compute.</p>
    </div>
  `,

  equations: [
    '\\dot K = s_K \\cdot \\text{GWP}',
    '\\dot L = g_L L',
    '\\dot{\\text{TFP}} = g_{\\text{TFP}} \\text{TFP}',
    '\\dot{HS} = F_C \\cdot \\text{GWP} \\cdot H_{t - h_d} - d_C \\cdot HS',
    'C = HS \\cdot S',
    '\\dot F_X = G_X \\cdot F_X',
  ],

  variables: [
    'K',
    'GWP',
    'L',
    'TFP',
    'HS',

    'H',
    'C',
    'S',

    'F_C',
    'F_{C,G}',
    'F_{C,H}',
    'F_{C,S}',
    'F_{C,T}',
    'F_{K,G}',
    'F_{K,H}',
    'F_{L,G}',
    'F_{L,H}',
    'F_{L,S}',
  ],

  parameters: [
    's_K', 
    'g_L',
    'g_{TFP}',
    'h_d',
    'd_C',

    'G_C',
    'G_{C,G}',
    'G_{C,H}',
    'G_{C,S}',
    'G_{C,T}',
    'G_{K,G}',
    'G_{K,H}',
    'G_{L,G}',
    'G_{L,H}',
    'G_{L,S}',
  ],
};

cards.push(reinvestmentCard);

let productionCard = {
  id: 'production',

  title: 'Production',

  abstract: 'The production module encompasses the production functions: goods and services, hardware R&D and software R&D. Each production function combines capital, labour, and inference compute to produce an output. The automation level determines the degree to which we can replace labour with inference compute in these production functions.',

  explanation: `
    <h4>Production of goods and services</h4>

    <p>A CES function gives the Gross World Product (GWP) in a given year:</p>

    <div>\\[GWP := TFP[\\alpha_G K_{G}^{\\rho_G} + (1-\\alpha_G) \\text{Cog}_G^{\\rho_G}]^{1/{\\rho_G}}\\]</div>

    <p>where \\(K_G\\) is the available capital dedicated to goods, \\(Cog_G\\) is the cognitive input, TFP is the total productivity factor, \\(\\rho_G\\) is the CES substitution and \\(\\alpha_G\\) is the weight assigned to capital.</p>

    <p>The cognitive input \\(\\text{Cog}_G\\) in each timestep is determined by the CES combination of the output of a non-AI compute task \\(C_{G,0}\\) and some cognitive tasks \\(T_{G,i} = L_{G,i} + \\eta_{G,i} C_{G,i}\\), where \\(i\\) indexes the different cognitive tasks, \\(L_{G,i}\\) is the labour allocated to task \\(i\\), \\(C_{G,i}\\) is the effective compute allocated to task \\(i\\) and \\(\\eta_i\\) is the efficiency with which we can replace labour by effective compute on task \\(i\\). The output of these tasks is combined as</p>

    <div>\\[Cog_G = [\\beta_{G,0} C_{G,0}^{\\psi_G} + \\beta_{G,1} T_{G,1}^{\\psi_G} + ... + \\beta_{G,N} T_{G,N}^{\\psi_G} ]^{1/{\\psi_G}}\\]</div>

    <p>where \\(\\beta_{G,i}\\) are the task-specific weights and \\(\\psi_G\\) is the CES substitution parameter.</p>

    <p>In each timestep of the model, we choose the labour per task \\(L_{G,1}, ..., L_{G,N}\\) and the effective compute per tasks \\(C_{G,0},...,C_{G,N}\\) to maximize the cognitive output, subject to the restrictions that 1) \\(L_{G,1} + ... + L_{G,N} = L_G\\) and \\(C_{G,0} + ... + C_{G,N} = C_G\\) and 2) that \\(C_{G,i} = 0\\) if the <a href="#automation-index">\\(A_G\\)</a> is below the task index \\(i\\). </p>


    <h4>Production of R&D input</h4>

    <p>The research input to hardware R&D is computed in the same way as the GWP</p>

    <div>\\[R_H := TFP[\\alpha_R K_{H}^{\\rho_R} + (1-\\alpha_R) \\text{Cog}_H^{\\rho_R}]^{1/{\\rho_R}}\\]</div>

    <div>\\[Cog_H := [\\beta_{R,0} C_{H,0}^{\\psi_R} + \\beta_{R,1} T_{H,1}^{\\psi_R} + ... + \\beta_{R,N} T_{H,N}^{\\psi_R} ]^{1/{\\psi_R}}\\]</div>

    <div>\\[T_{H,i} = L_{H,i} + \\eta_{R,i} C_{H,i}\\]</div>

    <p>For software R&D we assume no capital is needed to advance, and instead, the research input to software stems from a combination of the hardware stock \\(HS\\) (as a proxy of the physical compute available for experiments) and the cognitive output \\(Cog_S\\):</p>

    <div>\\[R_S := TFP[\\alpha_S HS^{\\zeta \\cdot \\rho_S} + (1-\\alpha_S) \\text{Cog}_S^{\\rho_S}]^{1/{\\rho_S}}\\]</div>

    <div>\\[Cog_S := [\\beta_{R,0} C_{S,0}^{\\psi_R} + \\beta_{R,1} T_{S,1}^{\\psi_R} + ... + \\beta_{R,N} T_{S,N}^{\\psi_R} ]^{1/{\\psi_R}}\\]</div>

    <div>\\[T_{S,i} = L_{S,i} + \\eta_{R,i} C_{S,i}\\]</div>

    <p>Note that the hardware stock HS is discounted by an exponential factor \\(\\zeta < 1\\). This represents a "stepping-on-toes" effect, accounting for diminishing returns to using more compute in experiments.</p>

    <p>Also, note that the hardware and software cognitive input production functions share the same parameters \\(\\psi_R, \\beta_R, \\eta_{R,i}\\), but have different inputs \\(L_{H,1} + ... + L_{H,N} = L_H\\), \\(C_{H,0} + ... + C_{H,N} = C_H\\), \\(L_{S,1} + ... + L_{S,N} = L_S\\), \\(C_{S,0} + ... + C_{S,N} = C_S\\).</p>

    <p>As with goods and services, we apply an automation restriction to hardware and software. Cognitive tasks can only be automated if their index \\(i\\) is below the R&D automation indicator \\(A_R\\). As before, we choose \\(L_{x,i}\\) and \\(C_{x,i}\\) to maximise cognitive output subject to the constraint that \\(C_{x,i} = 0\\) if \\(A_R < i\\).</p>
  `,

  equations: [
    'GWP := TFP[\\alpha_G K_{G}^{\\rho_G} + (1-\\alpha_G) \\text{Cog}_G^{\\rho_G}]^{1/{\\rho_G}}',
    'Cog_G = [\\beta_{G,0} C_{G,0}^{\\psi_G} + \\beta_{G,1} T_{G,1}^{\\psi_G} + ... + \\beta_{G,N} T_{G,N}^{\\psi_G} ]^{1/{\\psi_G}}',
    'T_{G,i} = L_{G,i} + \\eta_{G,i} C_{G,i}',

    'R_H := TFP[\\alpha_R K_{H}^{\\rho_R} + (1-\\alpha_R) \\text{Cog}_H^{\\rho_R}]^{1/{\\rho_R}}',
    'Cog_H := [\\beta_{R,0} C_{H,0}^{\\psi_R} + \\beta_{R,1} T_{H,1}^{\\psi_R} + ... + \\beta_{R,N} T_{H,N}^{\\psi_R} ]^{1/{\\psi_R}}',
    'T_{H,i} = L_{H,i} + \\eta_{R,i} C_{H,i}',

    'R_S := TFP[\\alpha_S HS^{\\zeta \\cdot \\rho_S} + (1-\\alpha_S) \\text{Cog}_S^{\\rho_S}]^{1/{\\rho_S}}',
    'Cog_S := [\\beta_{R,0} C_{S,0}^{\\psi_R} + \\beta_{R,1} T_{S,1}^{\\psi_R} + ... + \\beta_{R,N} T_{S,N}^{\\psi_R} ]^{1/{\\psi_R}}',
    'T_{S,i} = L_{S,i} + \\eta_{R,i} C_{S,i}',
  ],

  variables: [
    'GWP',
    'TFP',
    'K_G',
    'Cog_G',
    'T_{G,i}',
    'L_{G,i}',
    'C_{G,i}',

    'R_H',
    'Cog_H',
    'T_{H,i}',
    'L_{H,i}',
    'C_{H,i}',

    'R_S',
    'HS',
    'Cog_S',

    'T_{S,i}',
    'L_{S,i}',
    'C_{S,i}',
  ],

  parameters: [
    '\\alpha_G',
    '\\rho_G',
    '\\beta_{G,i}',
    '\\psi_G',
    '\\eta_{G,i}',

    '\\alpha_R',
    '\\rho_R',
    '\\beta_{R,i}',
    '\\psi_R',
    '\\eta_{R,i}',

    '\\alpha_S',
    '\\rho_S',
    '\\zeta',
  ],
};

cards.push(productionCard);

///////////////////////////////////////////////////////////////////////////////
//

let plt = new Plotter();

let js_params = transform_python_to_js_params(best_guess_parameters);
let sim = ftm.run_simulation(js_params);

// See https://atomiks.github.io/tippyjs/v6/plugins/#hideonesc
const hideOnEsc = {
  name: 'hideOnEsc',
  defaultValue: true,
  fn({hide}) {
    function onKeyDown(event) {
      if (event.keyCode === 27) {
        hide();
      }
    }

    return {
      onShow() {
        document.addEventListener('keydown', onKeyDown);
      },
      onHide() {
        document.removeEventListener('keydown', onKeyDown);
      },
    };
  },
};

function html(str) {
  let parentTag = 'div';
  if (str.trim().startsWith('<tr>')) {
    parentTag = 'tbody';
  }

  let tmp = document.createElement(parentTag);
  tmp.innerHTML = str.trim();
  let node = tmp.firstChild;
  return node;
}

function plot_vlines(sim, line_color = 'black', graph = null) {
  graph ||= plt;

  if (sim.rampup_start) {
    graph.axvline(sim.rampup_start, {
      linestyle: 'dotted',
      color: line_color,
      label: 'Wake-up',
    });
  }
              
  if (sim.rampup_mid) {
    graph.axvline(sim.rampup_mid, {
      linestyle: '-.',
      color: line_color,
      label: '20% automation',
    });
  }
              
  if (sim.timeline_metrics['automation_gns_100%']) {
    graph.axvline(sim.timeline_metrics['automation_gns_100%'], {
      linestyle: 'dashed',
      color: line_color,
      label: '100% automation',
    });
  }
}

function plot_variable(variable, container) {
  let t = sim.timesteps;
  let v = sim.get_thread(variable.thread)

  let crop_year = sim.timeline_metrics['automation_gns_100%'] + 5;
  let end_idx = (sim.timesteps[sim.timesteps.length-1] >= crop_year) ? nj.argmax(nj.gte(t, crop_year)) : t.length;

  t = t.slice(0, end_idx);
  v = v.slice(0, end_idx);

  plt.plot(t, v);

  plt.set_width(518);
  plt.set_height(350);

  plt.set_tooltip((x, ys) => {
    let y = ys[0];
    let content = `<span>Year: ${x.toFixed(1)} <br> ${variable.meaning}: ${y.toExponential(1)}</span>`;
    let node = html(content);
    MathJax.typeset([node]);
    return node;
  });

  plot_vlines(sim);
  plt.yscale(variable.yscale || 'log');

  let graph = plt.show(container);

  return graph;
}

// Deal with internal links

function followInternalLink(href) {
  // Is it inside a card?
  for (let card of cards) {
    let node = card.node.matches(href) ? card.node : card.node.querySelector(href);
    if (node) {
      // It is
      openCard(card);

      function isInLeftSection(node) {
        if (node == null) return false;
        if (node.classList.contains('left')) return true;
        return isInLeftSection(node.parentElement);
      }

      if (isInLeftSection(node)) {
        // Scroll inside the card
        let left = card.node.querySelector('.section.left');
        left.scrollTop = node.getBoundingClientRect().top;
      } else {
        node.scrollIntoView({
          behavior: 'smooth',
        });
      }
      return;
    }
  }

  // Is it a fold?
  for (let fold of appendixAccordion.folds) {
    if (fold.button.id == href.slice(1)) {
      openFold(fold);
      return;
    }
  }
}

function openFold(fold) {
  function scroll() {
    fold.header.scrollIntoView({
      behavior: 'smooth',
    });
  }

  appendixAccordion.once('fold:opened', (fold) => {
    scroll();
  });

  if (fold.expanded) {
    scroll();
  } else {
    fold.open({transition: false});
  }
}

function processInternalLinks(node) {
  let internalLinks = node.querySelectorAll('a[href^="#"]');
  for (let link of internalLinks) {
    link.addEventListener("click", (e) => {
      followInternalLink(link.getAttribute('href'));
      e.preventDefault();
    });
  }
}

function renderCard(card) {
  let template = `
    <div class="card">

      <div class="left section section-content">
        <h3 class="title"></h3>
        <div class="explanation"></div>
      </div>

      <div class="right">
        <div class="equations-box section">
          <div class="section-label">Key equations</div>
          <div class="equations section-content"></div>
        </div>

        <div class="variables-box section">
          <div class="section-label">Variables and parameters</div>
          <div class="variables section-content"></div>
        </div>

        <div class="parameters-box section" style="display: none">
          <div class="section-label">Parameter glossary and values</div>
          <div class="parameters section-content"></div>
        </div>
      </div>

    </div>
  `;

  let cardNode = html(template);

  cardNode.id = card.id;

  // Make the left section be the same height as the right one
  let left = cardNode.querySelector('.left');
  let right = cardNode.querySelector('.right');
  let resizeObserver = new ResizeObserver(function() {
    left.style.height = `${right.getBoundingClientRect().height}px`;
  });
  resizeObserver.observe(right);

  // Title
  cardNode.querySelector('.title').innerHTML = card.title;

  // Explanations
  let abstract = card.abstract.split('\n').filter(s => s != '').map(s => '<p>' + s + '</p>').join('\n');
  //let explanation = card.explanation.split('\n').filter(s => s != '').map(s => '<p>' + s + '</p>').join('\n');
  let explanation = card.explanation;
  cardNode.querySelector('.explanation').innerHTML = `<div class="abstract">${abstract}</div>` + '\n' + explanation;

  // Equations
  let equations = card.equations.map(s => '<p>\\[' + s + '\\]</p>').join('\n');
  cardNode.querySelector('.equations').innerHTML = equations;

  // Variables
  let variablesTable = html(`
    <table>
      <thead>
        <th>Variable</th>
        <th>Meaning</th>
        <th>Evolution in the best guess scenario</th>
      </thead>
      <tbody>
      </tbody>
    </table>
  `);

  for (let varName of card.variables) {
    let variable = variables[varName];

    let tr = html(`
      <tr>
        <td>${variable.repr || "\\(" + variable.name + "\\)"}</td>
        <td>${variable.meaning}</td>
        <td class="view-column"></td>
      </tr>
    `);

    if (variable.thread) {
      let icon = `
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-graph-up" viewBox="0 0 16 16">
        <path fill-rule="evenodd" d="M0 0h1v15h15v1H0V0Zm14.817 3.113a.5.5 0 0 1 .07.704l-4.5 5.5a.5.5 0 0 1-.74.037L7.06 6.767l-3.656 5.027a.5.5 0 0 1-.808-.588l4-5.5a.5.5 0 0 1 .758-.06l2.609 2.61 4.15-5.073a.5.5 0 0 1 .704-.07Z"/>
      </svg>
      `;

      let viewButton = html(`<div class="view-button-hitbox"><div class="view-button"><i>${icon}</i></div></div>`);
      let viewColumn = tr.querySelector('.view-column');
      //viewColumn.classList.add('view-button');
      viewColumn.appendChild(viewButton);
      let viewButtonHitbox = tr.querySelector('.view-button-hitbox');

      let graph;

      tippy(viewButton, {
        content: `<div class="plot-container"><div class="plot-title">${variable.meaning}</div></div>`,
        triggerTarget: viewButtonHitbox,
        trigger: 'click',
        duration: [10, 10],
        allowHTML: true,
        interactive: true,
        placement: 'right',
        appendTo: document.body,
        hideOnClick: false,
        theme: 'light-border',
        plugins: [hideOnEsc],
        maxWidth: '860px',
        onCreate: (instance) => {
          let plotContainer = instance.popper.querySelector('.plot-container');
          graph = plot_variable(variable, plotContainer);
        },
        onShow: (instance) => {
          tippy.hideAll();
        },
        onHidden: (instance) => {
          graph.hide_tooltip();
        },
      });
    }

    let tbody = variablesTable.querySelector('tbody');
    tbody.appendChild(tr);
  }

  cardNode.querySelector('.variables').appendChild(html('<h4>Variables</h4>'));
  cardNode.querySelector('.variables').appendChild(variablesTable);

  // Parameters

  let parametersTable = html(`
    <table>
      <thead>
        <th>Parameter</th>
        <th>Meaning</th>
        <!--<th>Best guess value</th>-->
      </thead>
      <tbody>
      </tbody>
    </table>
  `);

  for (let paramName of card.parameters) {
    let param = parameters[paramName];

    let tr = html(`
      <tr>
        <td>${param.repr || "\\(" + param.name + "\\)"}</td>
        <td>${param.meaning}</td>
        <!--<td>${param.notes}</td>-->
      </tr>
    `);

    let tbody = parametersTable.querySelector('tbody');
    tbody.appendChild(tr);
  }

  cardNode.querySelector('.variables').appendChild(html('<h4>Parameters</h4>'));
  //cardNode.querySelector('.parameters').appendChild(parametersTable);
  cardNode.querySelector('.variables').appendChild(parametersTable);

  setTimeout(function () {
    MathJax.typeset([cardNode])
  }, 0);

  for (let node of cardNode.querySelectorAll('[data-tooltip]')) {
    tippy(node, {
      content: node.dataset.tooltip,
      trigger: 'mouseenter click',
      duration: [10, 10],
      allowHTML: true,
      theme: 'light-border',
      plugins: [hideOnEsc],
      onCreate: (instance) => {
        MathJax.typeset([instance.popper]);
      },
    });
  }

  processInternalLinks(cardNode);

  let initialized = false;
  card.onOpen = () => {
    if (!initialized) {
      setTimeout(function () {
        processLearnMores(cardNode);
      }, 0);
      initialized = true;
    }
  };

  return cardNode;
}

function processLearnMores(node) {
  for (let learnMore of node.querySelectorAll('.learn-more')) {
    let contentWrapper = html('<div class="learn-more-content"></div>');
    while (learnMore.firstChild) {
      contentWrapper.appendChild(learnMore.firstChild);
    }

    let header = html('<div class="learn-more-header">Learn more</div>');
    learnMore.appendChild(header);
    learnMore.appendChild(contentWrapper);

    /*
    let height = contentWrapper.getBoundingClientRect().height;
    contentWrapper.style.maxHeight = 0;
    */
    contentWrapper.style.maxHeight = 0;
    learnMore.classList.add('closed');

    function updateMaxHeight() {
      if (learnMore.classList.contains('closed')) {
        contentWrapper.style.maxHeight = 0;
      } else {
        let curMaxHeight = contentWrapper.style.maxHeight;
        contentWrapper.style.maxHeight = '';
        let height = contentWrapper.getBoundingClientRect().height;
        contentWrapper.style.maxHeight = curMaxHeight;
        setTimeout(() => {
          contentWrapper.style.maxHeight = `${height}px`;
        }, 0);
      }
    }

    let prevWidth = null;
    let widthObserver = new ResizeObserver(function() {
      let curWidth = learnMore.getBoundingClientRect().width;
      if (prevWidth != curWidth) {
        prevWidth = curWidth;
        updateMaxHeight();
      }
    });
    widthObserver.observe(learnMore);

    header.addEventListener('click', () => {
      learnMore.classList.toggle('closed');
      updateMaxHeight();
    });
  }
}

let currentCard = null;

for (let card of cards) {
  let cardNode = renderCard(card);

  card.node = cardNode;

  let boxIds = card.boxes || [`${card.id}-box`];
  for (let boxId of boxIds) {
    let box = document.getElementById(boxId);
    box.addEventListener('click', () => {
      if (card == currentCard) {
        closeCards();
      } else {
        openCard(card);
      }
    });
  }
}

function closeCards() {
  let nodeContainer = document.querySelector('#node-container');
  while (nodeContainer.firstChild) nodeContainer.removeChild(nodeContainer.firstChild);
  currentCard = null;
  for (let box of document.querySelectorAll('svg .box')) {
    box.classList.remove('selected');
  }
}

function openCard(card) {
  closeCards();

  let nodeContainer = document.querySelector('#node-container');
  nodeContainer.appendChild(card.node);

  currentCard = card;

  for (let box of document.querySelectorAll('svg .box')) {
    box.classList.remove('selected');
  }

  let boxIds = card.boxes || [`${card.id}-box`];
  for (let boxId of boxIds) {
    let box = document.getElementById(boxId);
    box.classList.add('selected');
  }

  card.onOpen();
}

let appendixAccordion = new handorgel(document.querySelector('.appendices-container'), {
  multiSelectable: false,
});

document.querySelector('.appendices-container').classList.remove('invisible');

processInternalLinks(document);

if (location.hash) {
  followInternalLink(location.hash);
}

// Close tooltips when clicking outside them
document.body.addEventListener('mousedown', (e) => {
  function isInsideTooltipOrViewButton(node) {
    if (node == null) return false;
    if ('tippyRoot' in node.dataset) return true;
    if (node.classList.contains('view-button-hitbox')) return true;
    return isInsideTooltipOrViewButton(node.parentElement);
  }

  if (!isInsideTooltipOrViewButton(e.target)) {
    tippy.hideAll();
  }
});

///////////////////////////////////////////////////////////////////////////////
// Driver for easier development
//document.querySelector('#investment-box').dispatchEvent(new Event('click'));
//openCard(rndCard);
