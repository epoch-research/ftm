"use strict";

let cards = [];

let space_between_equations = '3pt';

let variables = {
  'K':       {meaning: 'Capital',                                                thread: 'capital', unit: '2022 USD'},

  'K_G':     {meaning: 'Capital dedicated to G&S',                               thread: 'goods.capital', unit: '2022 USD'},
  'Cog_G':   {meaning: 'Cognitive input to G&S',                                 thread: 'goods.cognitive_output', unit: '2022 USD'},
  'T_{G,i}': {meaning: 'Output of the \\(i\\)-th G&S task',                      },
  'C_{G,0}': {meaning: 'Output of the non-AI G&S compute tasks',                 thread: 'goods.compute_task_input[0]'},
  'C_{G,i}': {meaning: 'Compute allocated to the \\(i\\)-th G&S task',           },
  'L_{G,i}': {meaning: 'Labour allocated to the \\(i\\)-th G&S task',            },

  'R_H':     {meaning: 'Research input to hardware R&D',                         thread: 'hardware_performance.rnd_input', unit: '2022 USD'},
  'R_S':     {meaning: 'Research input to software R&D',                         thread: 'software.rnd_input', unit: '2022 USD'},

  'Cog_H':   {meaning: 'Cognitive input to hardware R&D',                        thread: 'hardware_rnd.cognitive_output', unit: '2022 USD'},
  'T_{H,i}': {meaning: 'Output of the \\(i\\)-th hardware R&D task',             },
  'C_{H,0}': {meaning: 'Output of the non-AI hardware R&D compute tasks',        thread: 'hardware_rnd.compute_task_input[0]'},
  'C_{H,i}': {meaning: 'Compute allocated to the \\(i\\)-th hardware R&D task',  },
  'L_{H,i}': {meaning: 'Labour allocated to the \\(i\\)-th hardware R&D task',   },

  'Cog_S':   {meaning: 'Cognitive input to software R&D',                        thread: 'software_rnd.cognitive_output', unit: '2022 USD'},
  'T_{S,i}': {meaning: 'Output of the \\(i\\)-th software R&D task',             },
  'C_{S,0}': {meaning: 'Output of the non-AI software R&D compute tasks',        thread: 'software_rnd.compute_task_input[0]'},
  'C_{S,i}': {meaning: 'Compute allocated to the \\(i\\)-th software R&D task',  },
  'L_{S,i}': {meaning: 'Labour allocated to the \\(i\\)-th software R&D task',   },

  'GWP':     {meaning: 'Gross world product',                                    thread: 'gwp', unit: '2022 USD'},

  'L':       {meaning: 'Labour',                                                 thread: 'labour', unit: 'years of human labour/year'},
  'TFP':     {meaning: 'Total factor productivity',                              thread: 'goods.tfp'},

  'HS':      {meaning: 'Hardware stock',                                         thread: 'hardware', unit: 'FLOP/year'},

  'H':       {meaning: 'Hardware efficiency',                                    thread: 'hardware_performance.v', unit: 'FLOP/year/$'},

  'C':       {meaning: 'Effective compute',                                      thread: 'compute', unit: '2022 FLOP/year'},
  'S':       {meaning: 'Software efficiency level',                              thread: 'software.v', unit: '2022 FLOP/FLOP'},

  'F_C':     {meaning: 'Fraction of GWP used to purchase new hardware',          thread: 'frac_gwp.compute.v'},
  'F_{C,G}': {meaning: 'Fraction of effective compute assigned to G&S',          thread: 'frac_compute.goods.v'},
  'F_{C,H}': {meaning: 'Fraction of effective compute assigned to hardware R&D', thread: 'frac_compute.hardware_rnd.v'},
  'F_{C,S}': {meaning: 'Fraction of effective compute assigned to software R&D', thread: 'frac_compute.software_rnd.v'},
  'F_{C,T}': {meaning: 'Fraction of effective compute assigned to training',     thread: 'frac_compute.training.v'},
  'F_{K,G}': {meaning: 'Fraction of capital assigned to G&S',                    thread: 'frac_capital.goods.v'},
  'F_{K,H}': {meaning: 'Fraction of capital assigned to hardware R&D',           thread: 'frac_capital.hardware_rnd.v'},
  'F_{L,G}': {meaning: 'Fraction of labor assigned to G&S',                      thread: 'frac_labour.goods.v'},
  'F_{L,H}': {meaning: 'Fraction of labor assigned to hardware R&D',             thread: 'frac_labour.hardware_rnd.v'},
  'F_{L,S}': {meaning: 'Fraction of labor assigned to software R&D',             thread: 'frac_labour.software_rnd.v'},

  'C_T':     {meaning: 'Largest training run',                                   thread: 'biggest_training_run', unit: '2022 FLOP'},

  'P_H':     {meaning: 'Penalty factor for hardware R&D',                        thread: 'hardware_rnd.ceiling_penalty'},
  'Q_H':     {meaning: 'Adjusted cumulative inputs to hardware R&D',             thread: 'hardware_performance.cumulative_rnd_input', unit: '2022 USD'},

  'P_S':     {meaning: 'Penalty factor for software R&D',                        thread: 'software_rnd.ceiling_penalty'},
  'Q_S':     {meaning: 'Adjusted cumulative inputs to software R&D',             thread: 'software.cumulative_rnd_input', unit: '2022 USD'},

  'A_G':     {meaning: 'Automation index for G&S',                               thread: 'goods.at',                                   yscale: 'linear'},
  'A_R':     {meaning: 'Automation index for R&D',                               thread: 'rnd.at',                                     yscale: 'linear'},
}

let parameters = {
  'g_L': {
    constant: 'labour_growth',
    meaning: 'Growth rate of labour',
    unit: '/year',
    justification: "<a href='https://www.google.com/search?q=world+population+growth+rate&ei=VdAOY_7TDqu00PEP6pOViAM&ved=0ahUKEwi-0qLTjPD5AhUrGjQIHepJBTEQ4dUDCA4&uact=5&oq=world+population+growth+rate&gs_lcp=Cgdnd3Mtd2l6EAMyCAgAEIAEELEDMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIICAAQgAQQyQMyBQgAEIAEOgcIABBHELADOgQIABBDSgUIPBIBMUoECEEYAEoECEYYAFCOBVjcCGDnCWgBcAF4AIABVogB0gKSAQE1mAEAoAEByAEIwAEB&sclient=gws-wiz'>Source</a>.",
  },

  'g_{TFP}': {
    constant: 'tfp_growth',
    meaning: 'Growth rate of TFP',
    unit: '/year',
    justification: "Average TFP growth over the last 20 years. <a href='https://docs.google.com/spreadsheets/d/1C-RUowD3Nwo51UF5ZeBjbeLwaw4HQ1o13KyJlhuXCcU/edit#gid=2116796644'>Source</a>.",
  },

  'h_d': {
    constant: 'hardware_delay',
    meaning: 'Hardware delay',
    unit: 'years',
    justification: 'Discussed <a href="https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#bookmark=id.2gnv7nk1tdrv">here</a>.',
  },

  'd_C': {
    constant: 'compute_depreciation',
    meaning: 'Hardware depreciation rate',
    unit: '/year',
    justification: 'Rough guess.',
  },

  '\\tau_{R,i}': {
    constant: sim => sim.consts.rnd.automation_training_flops.slice(1),
    graph: {
      indices: sim => nj.arange(1, sim.consts.rnd.automation_training_flops.length, 1),
      tooltip: (x, y) => `Task: ${x.toFixed(0)} <br> 2022 FLOP: ${y.toExponential(1)}`,
    },
    meaning: 'Training requirements for automation of the \\(i\\)-th R&D task',

    unit: '2022 FLOP',

    justification: 'Computed following the method described on the left, assuming training an AGI will require 1e36 FLOP (we\'re mostly anchoring to the Bio Anchors report, with an extra OOM to account for TAI being a lower bar than full automation) and the FLOP gap is 1e4 (see discussion in the <a href="https://docs.google.com/document/d/1os_4YOw6Xv33KjX-kR76D3kW1drkWRHKG2caeiEWzNs/edit#heading=h.ri0lo7o4pv24">summary</a> and <a href="https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.o4db3tcgrq28">full report</a>).',
  },

  '\\tau_{G,i}': {
    constant: sim => sim.consts.goods.automation_training_flops.slice(1),
    graph: {
      indices: sim => nj.arange(1, sim.consts.goods.automation_training_flops.length, 1),
      tooltip: (x, y) => `Task: ${x.toFixed(0)} <br> 2022 FLOP: ${y.toExponential(1)}`,
    },
    unit: '2022 FLOP',
    meaning: 'Training requirements for automation of the \\(i\\)-th G&S task',
    justification: 'Three times the requirements for R&D (see above). We have slightly lower requirements for R&D because it may be easier to gather data, and there may be fewer regulatory restrictions.',
  },

  's_K': {
    constant: 'investment_rate',
    meaning: 'Savings rate',
    unit: '/year',
    justification: 'See <a href="https://www.theglobaleconomy.com/rankings/savings/">here</a>. (Note that this parameter does not affect the dynamics of the model.)',
  },

  '\\alpha_G': {
    constant: 'goods.capital_task_weights[0]',
    meaning: 'Weight associated to capital in the G&S CES production function',
    justification: 'See <a href="#appendix-a">appendix A</a>.',
  },

  '\\beta_{G,i}': {
    constant: sim => `Task 0: 1 <br><br> Rest of tasks: <span class="no-break">${standard_format(sim.consts.goods.labour_task_weights[1])}</span>`,
    meaning: 'Task weights in the G&S cognitive CES function',
    justification: 'See <a href="#appendix-a">appendix A</a>.',
  },

  '\\rho_G': {
    constant: 'goods.capital_substitution',
    meaning: 'CES substitution for G&S',
    justification: 'Discussed <a href="https://docs.google.com/document/d/15EmltGq-kkiLO95AbvoB4ODVpyg26BgghvHBy1JDyZY/edit#heading=h.fo846mq256bx">here</a>.',
  },

  '\\psi_G': {
    constant: 'goods.labour_substitution',
    meaning: 'Substitution parameter in the G&S cognitive CES function',
    justification: 'Discussed <a href="https://docs.google.com/document/d/15EmltGq-kkiLO95AbvoB4ODVpyg26BgghvHBy1JDyZY/edit#heading=h.qazoq7gf2vgm">here</a>.',
  },

  '\\eta_{G,i}': {
    constant: sim => nj.div(1, sim.consts.goods.automation_runtime_flops.slice(1)),
    graph: {
      indices: sim => nj.arange(1, sim.consts.goods.automation_runtime_flops.length, 1),
      tooltip: (x, y) => `Task: ${x.toFixed(0)} <br> Efficiency: ${y.toExponential(1)}`,
    },
    meaning: 'Base compute-to-labour efficiency for the \\(i\\)-th G&S task <br> without runtime-training tradeoff',
    unit: 'years of human labour/2022 FLOP',
    justification: `
      Computed following the method explained at the end of the <a href="#automation-runtime-requirements">automation section</a>, assuming that running an AGI requires 1.67e16 FLOP/s and that the runtime FLOP gap is 10:

      <br>
      <br>

      <b>AGI runtime requirements</b>: We anchor on the Bio Anchors report (~1e16 FLOP/s), then we adjust upwards by 1 OOM to account for TAI being a lower bar than full automation and downwards by 6X to account for one-time advantages for AGI over humans in goods and services.

      <br>
      <br>

      <b>FLOP gap</b>: The spread of runtime requirements is smaller than the spread of training requirements for four reasons. First, a 10X increase in runtime compute typically corresponds to a 100X increase in training, e.g. for Chinchilla scaling. Secondly, increasing the horizon length of training tasks will increase training compute but not runtime. Thirdly, some of the "one time gains" for AGI over humans won't apply as much to pre-AGI systems; e.g. the benefits of thinking 100X faster are less for limited AIs that cannot learn independently over the course of weeks. Fourthly, a smaller spread is a hacky way to capture the fact it's harder to trade-off training compute for runtime compute today than it will be in the future.
    `,
  },

  '\\iota_{G,i}': {
    constant: sim => sim.consts.goods.automation_runtime_flops.slice(1),
    graph: {
      indices: sim => nj.arange(1, sim.consts.goods.automation_runtime_flops.length, 1),
      tooltip: (x, y) => `Task: ${x.toFixed(0)} <br> Requirements: ${y.toExponential(1)}`,
    },
    meaning: 'Base runtime requirements (no runtime-training tradeoff) for the \\(i\\)-th G&S task',
    unit: '2022 FLOP',
    justification: `
    Computed following the method explained on the left, assuming that running an AGI requires 1.67 FLOP/s and that the runtime FLOP gap is 10:

    <br>
    <br>

    <b>AGI runtime requirements</b>: We anchor on the Bio Anchors report (~1e16 FLOP/s), then we adjust upwards by 1 OOM to account for TAI being a lower bar than full automation and downwards by 6X to account for one-time advantages for AGI over humans in goods and services.

    <br>
    <br>

    <b>FLOP gap</b>: The spread of runtime requirements is smaller than the spread of training requirements for four reasons. First, a 10X increase in runtime compute typically corresponds to a 100X increase in training, e.g. for Chinchilla scaling. Secondly, increasing the horizon length of training tasks will increase training compute but not runtime. Thirdly, some of the "one time gains" for AGI over humans won't apply as much to pre-AGI systems; e.g. the benefits of thinking 100X faster are less for limited AIs that cannot learn independently over the course of weeks. Fourthly, a smaller spread is a hacky way to capture the fact it's harder to trade-off training compute for runtime compute today than it will be in the future.
    `,
  },

  '\\alpha_R': {
    constant: 'hardware_rnd.capital_task_weights[0]',
    meaning: 'Weight associated to capital in the hardware R&D CES production function',
    justification: 'See <a href="#appendix-a">appendix A</a>.',
  },

  '\\beta_{R,i}': {
    constant: 'hardware_rnd.capital_task_weights[0]',
    meaning: 'Task weights in the hardware and software R&D cognitive CES function',
    justification: 'See <a href="#appendix-a">appendix A</a>.',
  },

  '\\rho_R': {
    constant: 'hardware_rnd.capital_substitution',
    meaning: 'CES substitution for hardware R&D',
    justification: 'Discussed <a href="https://docs.google.com/document/d/15EmltGq-kkiLO95AbvoB4ODVpyg26BgghvHBy1JDyZY/edit#heading=h.o7tmwweugbb">here</a>.',
  },

  '\\psi_R': {
    constant: 'rnd.labour_substitution',
    meaning: 'Substitution parameter in the hardware and software R&D cognitive CES function',
    justification: 'Discussed <a href="https://docs.google.com/document/d/15EmltGq-kkiLO95AbvoB4ODVpyg26BgghvHBy1JDyZY/edit#heading=h.9t2n0pf04b2e">here</a>.',
  },

  '\\eta_{R,i}': {
    constant: sim => nj.div(1, sim.consts.rnd.automation_runtime_flops.slice(1)),
    graph: {
      indices: sim => nj.arange(1, sim.consts.rnd.automation_runtime_flops.length, 1),
      tooltip: (x, y) => `Task: ${x.toFixed(0)} <br> Efficiency: ${y.toExponential(1)}`,
    },
    meaning: 'Base compute-to-labour efficiency for the \\(i\\)-th hardware and software R&D task <br> without runtime-training tradeoff',
    unit: 'years of human labour/2022 FLOP',
    justification: `
      This is 100 times the compute-to-labour efficiency for G&S (see above).

      <br>
      <br>

      We estimate the one-time gains of AGIs over humans to be 60X in R&D vs 6X in G&S. This is a difference of 10X. 

      <br>
      <br>

      Then we add on another 10X because the model implicitly assumes that there are 0.8 million people doing software R&D in 2022, and 8 million people doing hardware R&D. (Because it multiplies the fraction of $ spent in these areas by the total labour force.) In fact I think the number of people working in these areas is ~10X less than this.
      `,
  },

  '\\iota_{R,i}': {
    constant: sim => sim.consts.rnd.automation_runtime_flops.slice(1),
    graph: {
      indices: sim => nj.arange(1, sim.consts.rnd.automation_runtime_flops.length, 1),
      tooltip: (x, y) => `Task: ${x.toFixed(0)} <br> Requirements: ${y.toExponential(1)} 2022 FLOP`,
    },
    meaning: 'Base runtime requirements (no runtime-training tradeoff) for the \\(i\\)-th R&D task',
    unit: '2022 FLOP',
    justification: `
      This is 100 times the base runtime requirements for G&S (see above).

      <br>
      <br>

      We estimate the one-time gains of AGIs over humans to be 60X in R&D vs 6X in G&S. This is a difference of 10X. 

      <br>
      <br>

      Then we add on another 10X because the model implicitly assumes that there are 0.8 million people doing software R&D in 2022, and 8 million people doing hardware R&D. (Because it multiplies the fraction of $ spent in these areas by the total labour force.) In fact I think the number of people working in these areas is ~10X less than this.
    `,
  },

  '\\alpha_S': {
    constant: 'software_rnd.capital_task_weights[0]',
    meaning: 'Weight associated to physical compute dedicated to experiments in the softwre R&D CES production function',
    justification: 'See <a href="#appendix-a">appendix A</a>.',
  },

  '\\rho_S': {
    constant: 'software_rnd.capital_substitution',
    meaning: 'CES substitution for software R&D',
    justification: 'See <a href="#appendix-c">appendix C</a>.',
  },

  '\\zeta': {
    constant: 'software_rnd.experiments_efficiency',
    meaning: 'Discounting factor for physical compute dedicated to experiments',
    justification: 'This parameter is chosen so that the effective R&D input from researchers and compute for experiments rose at the same rate over the last 10 years. (This is needed to keep their share of R&D constant in a CES production function. It also means we can change the importance of experiments without changing the retrodicted rate of recent progress.) In particular, we estimate that the number of researchers grew at 20% but physical compute grew at 50%. An exponent of 0.4 means that the effective input of compute for experiments also rose at 20%.',
  },

  'r_H': {
    constant: 'hardware_performance.returns',
    meaning: 'Efficiency of the returns to hardware R&D',
    justification: 'Discussed <a href="https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.9us1ymg9hau0">here</a>.',
  },

  'r_S': {
    constant: 'software.returns',
    meaning: 'Efficiency of the returns to software R&D',
    justification: 'Discussed <a href="https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.yzbcl83o650l">here</a>.',
  },

  'U_C': {
    constant: 'frac_gwp.compute.ceiling',
    meaning: 'Maximum fraction of GWP used to purchase new hardware',
    justification: 'Discussed <a href="https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#bookmark=id.rh3lt1q0f0hl">here</a>.',
  },

  'U_{C,H}': {
    constant: 'frac_compute.hardware_rnd.ceiling',
    meaning: 'Maximum fraction of effective compute assigned to hardware R&D',
    justification: 'If people anticipate an AI driven singularity, the demand for progress in AI R&D should become huge, such that the world allocates using a macroscopic of compute to AIs doing AI R&D.',
  },

  'U_{C,S}': {
    constant: 'frac_compute.software_rnd.ceiling',
    meaning: 'Maximum fraction of effective compute assigned to software R&D',
    justification: 'If people anticipate an AI driven singularity, the demand for progress in AI R&D should become huge, such that the world allocates using a macroscopic of compute to AIs doing AI R&D.'
  },

  'U_{C,T}': {
    constant: 'frac_compute.training.ceiling',
    meaning: 'Maximum fraction of effective compute assigned to training',
    justification: 'Discussed <a href="https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#bookmark=id.tz6v7gxroefr">here</a>.'
  },

  "U_{K,H}": {
    constant: 'frac_capital.hardware_rnd.ceiling',
    meaning: 'Maximum fraction of capital assigned to hardware R&D',
    justification: 'Discussed <span style="text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;"><a href="https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#bookmark=id.7ug8akmppx48" target="_blank">here</a></span>.'
  },

  "U_{L,H}": {
    constant: 'frac_labour.hardware_rnd.ceiling',
    meaning: 'Maximum fraction of labor assigned to hardware R&D',
    justification: 'As above.'
  },

  "U_{L,S}": {
    constant: 'frac_labour.software_rnd.ceiling',
    meaning: 'Maximum fraction of labor assigned to software R&D',
    justification: 'Discussed <span style="text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;"><a href="https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#bookmark=id.7ug8akmppx48" target="_blank">here</a></span>.'
  },

  "U_H": {
    constant: 'hardware_performance.ceiling',
    meaning: 'Maximum hardware efficiency',
    unit: 'FLOP/year/$',
    justification: 'Based on a rough estimate from a technical advisor. They guessed energy prices could eventually fall 10X from today to $0.01/kWh. And that (based on <span style="text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#56a3f1;"><a href="https://en.wikipedia.org/wiki/Landauer%27s_principle" target="_blank">Landauer’s limit</a></span>) you might eventually do 1e27 bit erasures per kWh. That implies 1e29 bit erasures per $. If we do 1 FLOP per bit, that\'s 1e29 FLOP/$. You could go get more FLOP/$ than this with reversible computing, at least 1e30 FLOP/$.<br/>The advisor separately estimated 1e24 FLOP/$ as the limit within the current paradigm (the value used in Bio Anchors).<br/>We are somewhat conservative and use the mid-point as our best guess, 1e27 FLOP/$.<br/>Lastly, we adjust these FLOP/$ down an OOM to get FLOP/year/$ (implying that we use chips for 10 years).'
  },

  "U_S": {
    constant: 'software.ceiling',
    meaning: 'Maximum software efficiency',
    unit: '2022&nbsp;FLOP/FLOP',
    justification: 'If training AGI currently requires 1e36 FLOP, but this can in the limit be reduced to human life-time learning FLOP of 1e24, that\'s 12 OOMs improvement. '
  },

  "H_0": {
    constant: 'initial.hardware_performance',
    meaning: 'Initial hardware efficiency',
    unit: 'FLOP/year/$',
    justification: 'The training of PaLM cost ~$10m, and used 3e24 FLOP. This implies 3e17 FLOP/$. If companies renting chips make back their money over 2 years then that corresponds to a <i>buyable</i> hardware performance of 1.5e17 FLOP/year/$. The initial hardware efficiency is calculated taking into account a hardware adoption delay of 1 year.'
  },

  "S_0": {
    constant: 'initial.software',
    meaning: 'Initial software efficiency',
    unit: '2022&nbsp;FLOP/FLOP',
    justification: 'Software efficiency is measured relative to the initial year (so it\'s 1 at the beginning of the simulation).'
  },


  "g_{K,H}": {
    constant: 'frac_capital.hardware_rnd.growth',
    meaning: 'Pre "wake-up" growth of the fraction of capital assigned to hardware R&D',
    unit: '/year',
    justification: 'Real $ investments in hardware R&D have <span style="text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;"><a href="https://docs.google.com/spreadsheets/d/1bGbzR0c3TqsRYTWS3s6Bysgh9ZOuKS1w6qH1SOI11iE/edit#gid=186138651" target="_blank">recently grown</a></span> at ~4%; subtracting out ~3% GWP growth implies ~1% growth in the fraction of GWP invested.'
  },

  "g_{L,H}": {
    constant: 'frac_labour.hardware_rnd.growth',
    meaning: 'Pre "wake-up" growth of the fraction of labor assigned to hardware R&D',
    unit: '/year',
    justification: 'As above.'
  },

  "g_{C,H}": {
    constant: 'frac_compute.hardware_rnd.growth',
    meaning: 'Pre "wake-up" growth of the fraction of effective compute assigned to hardware R&D',
    unit: '/year',
    justification: 'As above.'
  },


  "g_{L,S}": {
    constant: 'frac_labour.software_rnd.growth',
    meaning: 'Pre "wake-up" growth of the fraction of labor assigned to software R&D',
    unit: '/year',
    justification: 'Discussed <span style="text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;"><a href="https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.1v8m5dp6xefi" target="_blank">here</a></span>, calculations <span style="text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;"><a href="https://docs.google.com/spreadsheets/d/1qmiomnNLpjcWSaeT54KC1PH1hfi_jUFIkWszxJGVU5w/edit#gid=0" target="_blank">here</a></span>. We subtract out 1% population growth from 19% growth in number of reseachers (which is smaller than the 20% growth in real $).'
  },

  "g_{C,S}": {
    constant: 'frac_compute.software_rnd.growth',
    meaning: 'Pre "wake-up" growth of the fraction of effective compute assigned to software R&D',
    unit: '/year',
    justification: 'As above.'
  },

  "g_C": {
    constant: 'frac_gwp.compute.growth',
    meaning: 'Pre "wake-up" growth of the fraction of GWP used to purchase new hardware',
    unit: '/year',
    justification: 'Assumed to be equal to the growth rate post "wake up" (see below). Why? Demand for AI chips is smaller today than after ramp-up, pushing towards slower growth today. But growth today is from a smaller base, and can come from the share of GPUs growing as a fraction of semiconductor production (which won\'t be possible once it\'s already ~100% of production). We\'re assuming these effects cancel out.'
  },

  "g_{C,T}": {
    constant: 'frac_compute.training.growth',
    meaning: 'Pre "wake-up" growth of the fraction of effective compute assigned to training',
    unit: '/year',
    justification: 'This corresponds to the assumption that there will be a $4b training run in 2030, in line with Bio Anchors\' prediction. Discussed a little <span style="text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;"><a href="https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.yyref6x2mzxu" target="_blank">here</a></span>.'
  },


  "g'_{K,H}": {
    constant: 'frac_capital.hardware_rnd.growth_rampup',
    meaning: 'Post "wake-up" growth of the fraction of capital assigned to hardware R&D',
    unit: '/year',
    justification: 'Discussed <span style="text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;"><a href="https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.612idx97x187" target="_blank">here</a></span>. We substract out 3% annual GWP growth to calculate the growth in the fraction of GWP invested.'
  },

  "g'_{L,H}": {
    constant: 'frac_labour.hardware_rnd.growth_rampup',
    meaning: 'Post "wake-up" growth of the fraction of labor assigned to hardware R&D',
    unit: '/year',
    justification: 'As above.'
  },

  "g'_{C,H}": {
    constant: 'frac_compute.hardware_rnd.growth_rampup',
    meaning: 'Post "wake-up" growth of the fraction of effective compute assigned to hardware R&D',
    unit: '/year',
    justification: 'We assume a one-year doubling as compute can be easily reallocated to the now-extremely-lucrative field of AI R&D.'
  },


  "g'_{L,S}": {
    constant: 'frac_labour.software_rnd.growth_rampup',
    meaning: 'Post "wake-up" growth of the fraction of labor assigned to software R&D',
    unit: '/year',
    justification: 'Discussed <span style="text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;"><a href="https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.vi6088puv22e" target="_blank">here</a></span>. We substract out 3% annual GWP growth to calculate the growth in the <span style="font-style:italic;">fraction</span> of GWP invested.'
  },

  "g'_{C,S}": {
    constant: 'frac_compute.software_rnd.growth_rampup',
    meaning: 'Post "wake-up" growth of the fraction of effective compute assigned to software R&D',
    unit: '/year',
    justification: 'We assume a one-year doubling as compute can be easily reallocated to the now-extremely-lucrative field of AI R&D.'
  },

  "g'_C": {
    constant: 'frac_gwp.compute.growth_rampup',
    meaning: 'Post "wake-up" growth of the fraction of GWP used to purchase new hardware',
    unit: '/year',
    justification: 'Discussed <a href="https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.612idx97x187">here</a>. We substract out 3% annual GWP growth to calculate the growth in the fraction of GWP invested.'
  },

  "g'_{C,T}": {
    constant: 'frac_compute.training.growth_rampup',
    meaning: 'Post "wake-up" growth of the fraction of effective compute assigned to training',
    unit: '/year',
    justification: 'Discussed <span style="text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;"><a href="https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.5xk4lbt60vr0" target="_blank">here</a></span>.'
  },

  "\\lambda": {
    constant: 'rnd.parallelization_penalty',
    meaning: 'R&D parallelization penalty',
    justification: 'Economic models often use a value of 1. A prominent growth economist thought that values between 0.4 and 1 are reasonable. If you think adding new people really won\'t help much with AI progress, you could use a lower value. Besiroglu\'s <span style="text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;"><a href="https://tamaybesiroglu.com/papers/AreModels.pdf" target="_blank">thesis</a></span> cites estimates as low as 0.2 (p14).'
  },

  "\\delta": {
    constant: 'runtime_training_tradeoff',
    meaning: 'Efficiency of the runtime-training tradeoff',
    justification: ''
  },
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

    <div>\\[C_T = F_{C,T} \\cdot C \\cdot 1 \\, \\text{year} \\]</div>

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

    <p id="automation-runtime-requirements">We follow an identical procedure as for the training requirements to determine the runtime requirements for each task. We then define the compute-to-labour ratio of the cognitive tasks \\(\\eta_{x,i}\\) as the inverse of the runtime requirements. The result is that tasks that are easier to automate also require less effective compute to replace labour.</p>

    <p>Finally, we allow the automation of tasks where we haven't reached the corresponding effective compute automation threshold \\(\\tau_{G,i}\\) at the cost of a lower compute-to-labour efficiency \\(\\eta_{G,i}\\). This essentially allows AI developers to 'tradeoff' training and runtime effective compute – use excess runtime compute to make up for a smaller training run.</p>

    <p>In more detail, as long as the training compute \\(C_T\\) is within a threshold \\(\\tau_{G,i} / C_T < \\text{Max tradeoff}\\), the task is considered automated, with a compute-to-labour efficiency of:</p>

    <div>\\[\\eta_{G,i} = \\frac{1}{\\iota_{G,i} \\cdot\\left(\\frac{\\tau_{G,i}}{C_T} \\right)^\\delta}\\]</div>

    <p>where \\(\\iota_{G,i}\\) are the base runtime requirements and \\(\\delta > 1\\) is a parameter governing the efficiency of the tradeoff.</p>

    <p>Note that the efficiency increases smoothly beyond the base runtime requirements; runtime requirements never stop getting lower over time as more compute is invested in training larger AI systems.</p>
  `,

  equations: [
    'C_T = F_{C,T} \\cdot C \\cdot 1 \\, \\text{year}',

    `
    \\begin{cases}
        A_G = \\max_i \\{i : \\tau_{G,i} < C_T\\} \\\\
        \\eta_{G,i} = \\frac{1}{\\iota_{G,i} \\cdot\\left(\\frac{\\tau_{G,i}}{C_T} \\right)^\\delta}
    \\end{cases}
    `,

    `
    \\begin{cases}
      A_R = \\max_i \\{i : \\tau_{R,i} < C_T\\} \\\\
      \\eta_{R,i} = \\frac{1}{\\iota_{R,i} \\cdot\\left(\\frac{\\tau_{R,i}}{C_T} \\right)^\\delta}
    \\end{cases}
    `,
  ],

  variables: [
    'C_T',
    'F_{C,T}',
    'C',
    'A_G',
    'A_R',
    //'\\eta_{R,i}',
    //'\\eta_{G,i}',
    //'\\eta_{G,i}',
  ],

  parameters: [
    '\\delta',
    '\\tau_{G,i}',
    '\\iota_{G,i}',
    '\\tau_{R,i}',
    '\\iota_{R,i}',
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

      <p>Rearranging and integrating, we get \\(H = (\\delta/r)^{r} (\\int R_H^{\\lambda})^{r} = (\\delta/r)^{r} Q^{r}\\) taking the derivative and diving by \\(H\\) we arrive at the equation above.</p>

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

    'P_H = \\frac{\\log (U_H/H)}{\\log (U_H/H_0)}',

    '\\frac{\\dot S}{S} = {r_S \\cdot P_S} \\cdot\\frac{\\dot Q_S}{Q_S}',

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

  abstract: "The reinvestment module updates the available capital, labour, effective compute and total factor productivity (TFP) based on the model's output. Then it updates how these factors will be split between training new AIs, G&S production and R&D in the next timestep.",

  explanation: `
    <h4>Updating the factors of production</h4>

    <p>Capital varies as usual with the GWP</p>

    <div>\\[\\dot K = s_K \\cdot \\text{GWP}\\]</div>

    <p>where \\(s_K\\) is the savings rate<sup data-tooltip="We considered adding a capital depreciation factor \\(d_K\\) so that capital depreciates over time as \\(\\dot K = s_K GWP - d K\\). However, this didn't significantly affect the model's results, so we removed it." class="footnote" role="doc-noteref">1</sup>.</p>

    <p>Labour and TFP change exogenously at fixed growth rates \\(g_L\\) and \\(g_{TFP}\\):</p>

    <div>\\[\\dot L = g_L \\cdot L\\]</div>

    <div>\\[\\dot{\\text{TFP}} = g_{\\text{TFP}} \\cdot \\text{TFP}\\]</div>

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

      <p>The amount invested in training is assumed to be proportional to 'physical compute allocated to training' / FLOP/year/$; more specifically we calculate it as \\(C_T / (S \\cdot H_{t - h_d})\\), where \\(C_T = C \\cdot S_{C,T}\\) is the amount of effective compute allocated to training, \\(S\\) is the amount of effective FLOP per physical FLOP  and \\(H_{t - h_d}\\) is the physical FLOP/year/$ that is commercially available.</p>
       
      <p>So the variables for which the exceptions do not apply are the fractions of each input assigned to software R&D and hardware R&D, and the fraction of GDP spent on compute.</p>
    </div>
  `,

  equations: [
    '\\dot K = s_K \\cdot \\text{GWP}',

    '\\dot L = g_L \\cdot L',

    '\\dot{\\text{TFP}} = g_{\\text{TFP}} \\cdot \\text{TFP}',

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

    "g_{K,H}",
    "g_{L,H}",
    "g_{C,H}",
    "g_{L,S}",
    "g_{C,S}",
    "g_C",
    "g_{C,T}",

    "g'_{K,H}",
    "g'_{L,H}",
    "g'_{C,H}",
    "g'_{L,S}",
    "g'_{C,S}",
    "g'_C",
    "g'_{C,T}",
  ],
};

cards.push(reinvestmentCard);

let productionCard = {
  id: 'production',

  title: 'Production',

  abstract: 'The production module encompasses the production functions: goods and services, hardware R&D and software R&D. Each production function combines capital, labour, and effective compute to produce an output. The automation level determines the degree to which we can replace labour with inference compute in these production functions.',

  explanation: `
    <h4>Production of goods and services</h4>

    <p>A CES function gives the Gross World Product (GWP) in a given year:</p>

    <div>\\[GWP := TFP \\cdot [\\alpha_G \\cdot K_{G}^{\\rho_G} + (1-\\alpha_G) \\cdot \\text{Cog}_G^{\\rho_G}]^{1/{\\rho_G}}\\]</div>

    <p>where \\(K_G\\) is the available capital dedicated to goods, \\(Cog_G\\) is the cognitive input, TFP is the total productivity factor, \\(\\rho_G\\) is the CES substitution and \\(\\alpha_G\\) is the weight assigned to capital.</p>

    <p>The cognitive input \\(\\text{Cog}_G\\) in each timestep is determined by the CES combination of the output of a non-AI compute task \\(C_{G,0}\\) and some cognitive tasks \\(T_{G,i} = L_{G,i} + \\eta_{G,i} \\cdot C_{G,i}\\), where \\(i\\) indexes the different cognitive tasks, \\(L_{G,i}\\) is the labour allocated to task \\(i\\), \\(C_{G,i}\\) is the effective compute allocated to task \\(i\\) and \\(\\eta_i\\) is the efficiency with which we can replace labour by effective compute on task \\(i\\). The output of these tasks is combined as</p>

    <div>\\[Cog_G = [\\beta_{G,0} \\cdot C_{G,0}^{\\psi_G} + \\beta_{G,1} \\cdot T_{G,1}^{\\psi_G} + ... + \\beta_{G,N} \\cdot T_{G,N}^{\\psi_G} ]^{1/{\\psi_G}}\\]</div>

    <p>where \\(\\beta_{G,i}\\) are the task-specific weights and \\(\\psi_G\\) is the CES substitution parameter.</p>

    <p>In each timestep of the model, we choose the labour per task \\(L_{G,1}, ..., L_{G,N}\\) and the effective compute per tasks \\(C_{G,0},...,C_{G,N}\\) to maximize the cognitive output, subject to the restrictions that 1) \\(L_{G,1} + ... + L_{G,N} = L_G\\) and \\(C_{G,0} + ... + C_{G,N} = C_G\\) and 2) that \\(C_{G,i} = 0\\) if the <a href="#automation-index">\\(A_G\\)</a> is below the task index \\(i\\). </p>


    <h4>Production of R&D input</h4>

    <p>The research input to hardware R&D is computed in the same way as the GWP</p>

    <div>\\[R_H := TFP \\cdot [\\alpha_R \\cdot K_{H}^{\\rho_R} + (1-\\alpha_R) \\cdot \\text{Cog}_H^{\\rho_R}]^{1/{\\rho_R}}\\]</div>

    <div>\\[Cog_H := [\\beta_{R,0} \\cdot C_{H,0}^{\\psi_R} + \\beta_{R,1} \\cdot T_{H,1}^{\\psi_R} + ... + \\beta_{R,N} \\cdot T_{H,N}^{\\psi_R} ]^{1/{\\psi_R}}\\]</div>

    <div>\\[T_{H,i} = L_{H,i} + \\eta_{R,i} \\cdot C_{H,i}\\]</div>

    <p>For software R&D we assume no capital is needed to advance, and instead, the research input to software stems from a combination of the hardware stock \\(HS\\) (as a proxy of the physical compute available for experiments) and the cognitive output \\(Cog_S\\):</p>

    <div>\\[R_S := TFP \\cdot [\\alpha_S HS^{\\zeta \\cdot \\rho_S} + (1-\\alpha_S) \\cdot \\text{Cog}_S^{\\rho_S}]^{1/{\\rho_S}}\\]</div>

    <div>\\[Cog_S := [\\beta_{R,0} \\cdot C_{S,0}^{\\psi_R} + \\beta_{R,1} \\cdot T_{S,1}^{\\psi_R} + ... + \\beta_{R,N} \\cdot T_{S,N}^{\\psi_R} ]^{1/{\\psi_R}}\\]</div>

    <div>\\[T_{S,i} = L_{S,i} + \\eta_{R,i} \\cdot C_{S,i}\\]</div>

    <p>Note that the hardware stock HS is discounted by an exponential factor \\(\\zeta < 1\\). This represents a "stepping-on-toes" effect, accounting for diminishing returns to using more compute in experiments.</p>

    <p>Also, note that the hardware and software cognitive input production functions share the same parameters \\(\\psi_R, \\beta_R, \\eta_{R,i}\\), but have different inputs \\(L_{H,1} + ... + L_{H,N} = L_H\\), \\(C_{H,0} + ... + C_{H,N} = C_H\\), \\(L_{S,1} + ... + L_{S,N} = L_S\\), \\(C_{S,0} + ... + C_{S,N} = C_S\\).</p>

    <p>As with goods and services, we apply an automation restriction to hardware and software. Cognitive tasks can only be automated if their index \\(i\\) is below the R&D automation indicator \\(A_R\\). As before, we choose \\(L_{x,i}\\) and \\(C_{x,i}\\) to maximise cognitive output subject to the constraint that \\(C_{x,i} = 0\\) if \\(A_R < i\\).</p>
  `,

  equations: [
    `
    \\begin{cases}
      GWP := TFP \\cdot [\\alpha_G \\cdot K_{G}^{\\rho_G} + (1-\\alpha_G) \\cdot \\text{Cog}_G^{\\rho_G}]^{1/{\\rho_G}} \\\\
      Cog_G = [\\beta_{G,0} \\cdot C_{G,0}^{\\psi_G} + \\beta_{G,1} \\cdot T_{G,1}^{\\psi_G} + ... + \\beta_{G,N} \\cdot T_{G,N}^{\\psi_G} ]^{1/{\\psi_G}} \\\\
      T_{G,i} = L_{G,i} + \\eta_{G,i} \\cdot C_{G,i}
    \\end{cases}
    `,

    `
    \\begin{cases}
      R_H := TFP \\cdot [\\alpha_R \\cdot K_{H}^{\\rho_R} + (1-\\alpha_R) \\cdot \\text{Cog}_H^{\\rho_R}]^{1/{\\rho_R}} \\\\
      Cog_H := [\\beta_{R,0} \\cdot C_{H,0}^{\\psi_R} + \\beta_{R,1} \\cdot T_{H,1}^{\\psi_R} + ... + \\beta_{R,N} \\cdot T_{H,N}^{\\psi_R} ]^{1/{\\psi_R}} \\\\
      T_{H,i} = L_{H,i} + \\eta_{R,i} \\cdot C_{H,i}
    \\end{cases}
    `,

    `
    \\begin{cases}
      R_S := TFP \\cdot [\\alpha_S \\cdot HS^{\\zeta \\cdot \\rho_S} + (1-\\alpha_S) \\cdot \\text{Cog}_S^{\\rho_S}]^{1/{\\rho_S}} \\\\
      Cog_S := [\\beta_{R,0} \\cdot C_{S,0}^{\\psi_R} + \\beta_{R,1} \\cdot T_{S,1}^{\\psi_R} + ... + \\beta_{R,N} \\cdot T_{S,N}^{\\psi_R} ]^{1/{\\psi_R}} \\\\
      T_{S,i} = L_{S,i} + \\eta_{R,i} \\cdot C_{S,i}
    \\end{cases}
    `,
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
    '\\beta_{G,i}',
    '\\rho_G',
    '\\psi_G',
    '\\eta_{G,i}',

    '\\alpha_R',
    '\\beta_{R,i}',
    '\\rho_R',
    '\\psi_R',
    '\\eta_{R,i}',

    '\\alpha_S',
    '\\rho_S',
    '\\zeta',
  ],
};

cards.push(productionCard);
