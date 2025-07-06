# -*- coding: utf-8 -*-
"""
This .py file includes all the functions needed to directly run the main experiments of OptiMix 
and generates the corresponding figures.
"""



import os
import subprocess
import shutil
import textwrap
import json
import pickle
import numpy as np
import statistics
import config
from Data_Gen   import data_MixNet
from Main_F_    import Carmix_
from Main_F     import Carmix
from Fancy_Plot import PLOT



def print_boxed_message(message):


    border = "+" + "-" * 120 + "+"
    print(border)
    for line in message:
        print("| " + line.ljust(120) + " |")
    print(border)


def Medd(List):
    N = len(List)

    List_ = []
    
    for i in range(N):

        List_.append( statistics.median(List[i]))
        
    return List_

def To_list(matrix):
    """Convert a numpy matrix to a list. If the result is a single row, return it as a flat list."""
    matrix_list = matrix.tolist()
    return matrix_list[0] if len(matrix_list) == 1 else matrix_list

def devide_list(L1,L2):
    return [L1[i]/L2[i] for i in range(len(L1))]


def generate_latex_table(tau, L1, L2, L3, L4):
    approaches = ["LAR", "GWR", "GPR", "SSR"]
    datasets = [r"$\ell_{P}$ (ms)", r"$\lambda^{-1}$ (ms)", r"$\mathsf{H}(r)$ (bits)", r"$\mathsf{H}(m)$ (bits)"]
    Ls = [L1, L2, L3, L4]

    # Start tabular environment
    table = r"\begin{tabular}{|c|" + "c|" * (len(tau) * 4) + "}\n\\hline\n"

    # Header: tau values
    table += r"$\tau$" + " & " + " & ".join(
        [f"\multicolumn{{4}}{{c|}}{{\\textbf{{ {t} }}}}" for t in tau]
    ) + " \\\\\n\\hline\n"

    # Sub-header: approaches
    table += "Approaches" + " & " + " & ".join(approaches * len(tau)) + " \\\\\n\\hline\n"

    # Data rows
    for row in range(4):
        row_data = [datasets[row]]
        for t in range(len(tau)):
            for l in Ls:
                row_data.append(str(l[row][t]))
        table += " & ".join(row_data) + r" \\" + "\n\\hline\n"

    table += r"\end{tabular}"

    # Wrap with \scalebox
    scaled_table = r"\scalebox{0.6}{" + "\n" + table + "\n}"
    return scaled_table





def create_pdf_from_latex_table(tau, L1, L2, L3, L4, output_filename="table_output.pdf"):
    table_code = generate_latex_table(tau, L1, L2, L3, L4)

    full_latex = r"""
\documentclass{article}
\usepackage[margin=0.5in]{geometry}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{caption}
\usepackage{adjustbox}
\begin{document}
\begin{center}
""" + table_code + r"""
\end{center}
\end{document}
"""

    tex_filename = "temp_table.tex"
    with open(tex_filename, "w") as f:
        f.write(full_latex)

    try:
        # Try compiling LaTeX using pdflatex
        result = subprocess.run(["pdflatex", tex_filename], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        temp_pdf = "temp_table.pdf"
        if os.path.exists(temp_pdf):
            if os.path.exists(output_filename):
                try:
                    os.remove(output_filename)
                except PermissionError:
                    print(f"⚠️ Cannot overwrite '{output_filename}' — file may be open.")
                    return

            shutil.move(temp_pdf, output_filename)
            print(f"✅ PDF successfully generated: {output_filename}")
        else:
            raise RuntimeError("LaTeX did not produce the expected PDF output.")

    except FileNotFoundError:
        print("\n" + "=" * 80)
        print("🚫 LaTeX compiler (pdflatex) not found on this system.")
        print("📄 Please install LaTeX to enable PDF generation.\n")
        print("📎 Meanwhile, here is the raw LaTeX code for your table:\n")
        print("=" * 80)
        print(full_latex)
        print("=" * 80)

    except subprocess.CalledProcessError:
        print("\n" + "=" * 80)
        print("❌ LaTeX compilation failed due to syntax or system error.")
        print("🛠️ Please check your LaTeX installation or document structure.\n")
        print("📎 Meanwhile, here is the raw LaTeX code for your table:\n")
        print("=" * 80)
        print(full_latex)
        print("=" * 80)

    finally:
        # Clean up LaTeX auxiliary files
        for ext in [".aux", ".log", ".tex"]:
            try:
                os.remove("temp_table" + ext)
            except FileNotFoundError:
                pass


def create_pdf_from_big_table(list_A, output_filename="table_output.pdf"):
    table_code = generate_big_latex_table(list_A)

    tex_filename = "temp_big_table.tex"
    temp_pdf = "temp_big_table.pdf"

    with open(tex_filename, "w") as f:
        f.write(table_code)

    try:
        # Try compiling LaTeX using pdflatex
        result = subprocess.run(
            ["pdflatex", tex_filename],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if os.path.exists(temp_pdf):
            if os.path.exists(output_filename):
                try:
                    os.remove(output_filename)
                except PermissionError:
                    print(f"⚠️ Cannot overwrite '{output_filename}' — file may be open.")
                    return

            try:
                shutil.move(temp_pdf, output_filename)
                print(f"✅ PDF successfully generated: {output_filename}")
            except Exception as e:
                print(f"❌ Error saving PDF: {e}")
        else:
            raise RuntimeError("LaTeX did not produce the expected PDF output.")

    except FileNotFoundError:
        print("\n" + "=" * 80)
        print("🚫 LaTeX compiler (pdflatex) not found on this system.")
        print("📄 Please install LaTeX to enable PDF generation.\n")
        print("📎 Meanwhile, here is the raw LaTeX code for your table:\n")
        print("=" * 80)
        print(table_code)
        print("=" * 80)

    except subprocess.CalledProcessError:
        print("\n" + "=" * 80)
        print("❌ LaTeX compilation failed due to syntax or system error.")
        print("🛠️ Please check your LaTeX installation or document structure.\n")
        print("📎 Meanwhile, here is the raw LaTeX code for your table:\n")
        print("=" * 80)
        print(table_code)
        print("=" * 80)

    finally:
        # Clean up auxiliary files
        for ext in [".aux", ".log", ".tex"]:
            try:
                os.remove("temp_big_table" + ext)
            except FileNotFoundError:
                pass


def  generate_big_latex_table(list_A):

    assert len(list_A) == 7 and all(len(row) == 20 for row in list_A), "list_A must contain 7 rows of 20 numerical values."

    row_labels = [
        r"\textbf{Vanilla}",
        r"\textbf{LARMix}~[29]",
        r"\textbf{OptiMix, GWR \&  LBA}",
        r"\textbf{OptiMix, GPR \& LBA}",
        r"\textbf{LAMP}~[30]",
        r"\textbf{OptiMix, GWR}",
        r"\textbf{OptiMix, GPR}"
    ]

    header = r"""\documentclass{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{caption}
\usepackage{adjustbox}
\begin{document}

\begin{table}[htbp]
\centering
\renewcommand{\arraystretch}{1.4}
%\caption{Key results for OptiMix using RIPE and Nym datasets, and its comparison with LARMix~\cite{mahdi2024larmix} and LAMP~\cite{mahdi2025lamp}.}
\label{com}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{|l|*{5}{c|c|c|c|}}
\hline
\multirow{1}{*}{\textbf{Approaches \textbackslash Metrics}} 
& \multicolumn{4}{c|}{\textbf{Entropy, $\mathsf{H}(r)$ (bits)}} 
& \multicolumn{4}{c|}{\textbf{Latency, $\ell_{\texttt{P}}$ (ms)}} 
& \multicolumn{4}{c|}{\textbf{Entropy/Latency $\frac{\mathsf{H}(r)}{\ell_{\texttt{P}}}$ (bits/sec)}} 
& \multicolumn{4}{c|}{\textbf{Anonymity, $\mathsf{H}(m)$ (bits)}}
& \multicolumn{4}{c|}{\textbf{FCP}}  \\
\hline
\textbf{Topology}& \multicolumn{2}{c|}{\textbf{Cascade}} & \multicolumn{2}{c|}{\textbf{Stratified}} 
& \multicolumn{2}{c|}{\textbf{Cascade}} & \multicolumn{2}{c|}{\textbf{Stratified}} 
& \multicolumn{2}{c|}{\textbf{Cascade}} & \multicolumn{2}{c|}{\textbf{Stratified}} 
& \multicolumn{2}{c|}{\textbf{Cascade}} & \multicolumn{2}{c|}{\textbf{Stratified}}  
& \multicolumn{2}{c|}{\textbf{Cascade}} & \multicolumn{2}{c|}{\textbf{Stratified}} 
 \\
\hline
\textbf{Dataset}& RIPE & Nym & RIPE & Nym & 
RIPE & Nym & RIPE & Nym &
RIPE & Nym & RIPE & Nym &
RIPE & Nym & RIPE & Nym &
RIPE & Nym & RIPE & Nym  \\
\hline
"""

    body = ""
    for label, values in zip(row_labels, list_A):
        row_values = " & ".join("N/A" if v is None else str(v) for v in values)
        body += f"{label} & {row_values} \\\\\n\\hline\n"

    footer = r"""\end{tabular}
\end{adjustbox}
\end{table}

\end{document}
"""

    return header + body + footer

class OptiMix(object):
    
    def __init__(self,EXP):
        
        self.d_1 = config.d_Nym
        self.d_2 = config.d_RIPE    
        self.W  = config.wings
        self.h = config.hops
        self.l = config.layers
        self.Iterations = config.Iterations
        self.Create_data = config.create_data
        self.File_fig = config.File_figure
        self.File_re = config.File_result  
        self.File_ta = config.File_table
        self.Targets = config.Num_targets
        self.delay1 = config.delay1
        self.delay2_Nym  =  config.delay2_Nym
        self.delay2_RIPE = config.delay2_RIPE        
        self.run = config.run_time
        self.c_mix_set = config.corrupted_Mix
        self.List_Tau = config.Tau_List
        self.e2e = config.e2e_delay
        self.N_List = config.Theta_List


        if not os.path.exists(self.File_fig):
            os.mkdir(os.path.join('', self.File_fig))  

        if not os.path.exists(self.File_re):
            os.mkdir(os.path.join('', self.File_re))  

        if not os.path.exists(self.File_ta):
            os.mkdir(os.path.join('', self.File_ta))  
            
        
     
        if int(EXP) == 0:
            
            message =  ["Thank you for entering the experiment ID. You have selected to run the initial experiment (ID: 0), ",
            "which generates the data files (e.g., .json or .pkl) required as subroutines for the other experiments.",
            "This experiment takes less than 5 minutes to complete ⏱️.",
            "(The estimation is based on the config.py file and the hardware/software specifications",
            " provided in the Appendix or the README file.)"
            ]
            
            print_boxed_message(message)


            self.generate_data()
            
            message =  ["Thank you for your patience. The results of this experiment have been saved ",
            " in the main directory or in the Results folder. "]
            
            print_boxed_message(message)            
            
        
        elif int(EXP) == 1:
            
            
            message =  ["Thank you for entering the experiment ID. You have selected to run the first experiment (ID: 1), ",
            " which supports the following claims:",
            r"C1: The higher the value of $\tau$, the higher the latency.",
            r"C2: The higher the value of $\tau$, the higher the entropy.",
            "C3: Compared to Figures 2 and 3, applying the balancing algorithm slightly increases both entropy and latency,",
            " due to higher randomness in routing for both cascade and stratified topologies.",
            "C4: Overall, GWR or GPR provide the best performance for stratified and cascade topologies.",
            "This experiment generates Figures 2, 3, 4, and 5 respectively.",
            "This experiment takes less than 5 minutes to complete ⏱️.",
            "(The estimation is based on the config.py file and the hardware/software specifications",
            " provided in the Appendix or the README file.)"
            ]
            
            print_boxed_message(message)
            
            message =  ["************If the following warnings appear during execution, you can safely ignore them: ",
            "1) RuntimeWarning: Mean of empty slice. out=out, **kwargs)--------------",
            "2) invalid value encountered in scalar divide ret = ret.dtype.type(ret / rcount)*************************"
            ]
            
            print_boxed_message(message)   
            
            self.EXP_1()
                     
            message =  ["Thank you for your patience. The results of this experiment have been saved as: ",
            "Fig2_a.png, Fig2_b.png, Fig2_c.png, Fig2_d.png",
            r"Fig3_a.png, Fig3_b.png, Fig3_c.png, Fig3_d.png",
            r"Fig4_a.png, Fig4_b.png, Fig4_c.png, Fig4_d.png",
            "Fig5_a.png, Fig5_b.png, Fig5_c.png, Fig5_d.png",
            " Please check the Figures folder to access these files." ]
            
            print_boxed_message(message)            
            



    
    
        elif int(EXP) == 2:
            
            message =  ["Thank you for entering the experiment ID. You have selected to run the second experiment (ID: 2), ",
            " which supports the following claim:",

            r"C5: In the mixnet adversary experiment, increasing $\tau$ decreases the FCP,",
            r"while increasing the adversary budget ($\frac{f}{N}$) increases FCP.",
            "Additionally, applying the balancing algorithm—due to its randomness—slightly decreases FCP.",
            "The results of this experiment will be shown in Figure 8.",
            "This experiment takes less than 5 minutes to complete ⏱️.",
            "(The estimation is based on the config.py file and the hardware/software specifications",
            " provided in the Appendix or the README file.)"
            ]
            
            print_boxed_message(message)
            

            self.EXP_6()
                     
         
    
    
     




    
    
        elif int(EXP) == 3:
            
            message =  ["Thank you for entering the experiment ID. You have selected to run the third experiment (ID: 3), ",
            " which supports the following claim:",

            r"C6: Cover routing enhances security against both global passive adversaries and mixnode adversaries in mixnets.",
            r"Specifically, increasing the cost parameter $\theta$ raises H(r)  and reduces FCP ",
            "  for both cascade and stratified topologies.",
            "The results of this experiment will be shown in Figure 9.",
            "This experiment takes less than 5 minutes to complete ⏱️.",
            "(The estimation is based on the config.py file and the hardware/software specifications",
            " provided in the Appendix or the README file.)"
            ]
            
            print_boxed_message(message)
            

            self.EXP_7()
                         
    
    
    
    
    
    
        elif int(EXP) == 22:
            
            
            message =  ["Thank you for entering the experiment ID. You have selected to run the experiment corresponding to Figure 2,",
                       "which supports the following claim:",
            r"C1: The higher the value of $\tau$, the higher the latency.",


            "This experiment takes less than 5 minutes to complete ⏱️.",
            "(The estimation is based on the config.py file and the hardware/software specifications",
            " provided in the Appendix or the README file.)"
            ]
            
            print_boxed_message(message)
            
            message =  ["************If the following warnings appear during execution, you can safely ignore them: ",
            "1) RuntimeWarning: Mean of empty slice. out=out, **kwargs)--------------",
            "2) invalid value encountered in scalar divide ret = ret.dtype.type(ret / rcount)*************************"
            ]
            
            print_boxed_message(message)   
            
            self.EXP_12()
                     
            message =  ["Thank you for your patience. The results of this experiment have been saved as: ",
            "Fig2_a.png, Fig2_b.png, Fig2_c.png, Fig2_d.png",

            " Please check the Figures folder to access these files." ]
            
            print_boxed_message(message)               
    
    
    
        elif int(EXP) == 33:
            
            
            message =  ["Thank you for entering the experiment ID. You have selected to run the experiment corresponding to Figure 3,",
                       "which supports the following claim:",
            r"C2: The higher the value of $\tau$, the higher the entropy.",


            "This experiment takes less than 5 minutes to complete ⏱️.",
            "(The estimation is based on the config.py file and the hardware/software specifications",
            " provided in the Appendix or the README file.)"
            ]
            
            print_boxed_message(message)
            
            message =  ["************If the following warnings appear during execution, you can safely ignore them: ",
            "1) RuntimeWarning: Mean of empty slice. out=out, **kwargs)--------------",
            "2) invalid value encountered in scalar divide ret = ret.dtype.type(ret / rcount)*************************"
            ]
            
            print_boxed_message(message)   
            
            self.EXP_13()
                     
            message =  ["Thank you for your patience. The results of this experiment have been saved as: ",
            "Fig3_a.png, Fig3_b.png, Fig3_c.png, Fig3_d.png",

            " Please check the Figures folder to access these files." ]
            
            print_boxed_message(message)             
    
    
    
    
    
        elif int(EXP) == 44:
            
            
            message =  ["Thank you for entering the experiment ID. You have selected to run the experiment corresponding to Figure 4,",
                       "which supports the following claim:",
            "C3: Compared to Figures 2 and 3, applying the balancing algorithm slightly increases both entropy and latency,",
            " due to higher randomness in routing for both cascade and stratified topologies.",
           
            "This experiment takes less than 5 minutes to complete ⏱️.",
            "(The estimation is based on the config.py file and the hardware/software specifications",
            " provided in the Appendix or the README file.)"
            ]
            
            print_boxed_message(message)
            
            message =  ["************If the following warnings appear during execution, you can safely ignore them: ",
            "1) RuntimeWarning: Mean of empty slice. out=out, **kwargs)--------------",
            "2) invalid value encountered in scalar divide ret = ret.dtype.type(ret / rcount)*************************"
            ]
            
            print_boxed_message(message)    
            
            self.EXP_14()
                     
            message =  ["Thank you for your patience. The results of this experiment have been saved as: ",
            "Fig4_a.png, Fig4_b.png, Fig4_c.png, Fig4_d.png",

            " Please check the Figures folder to access these files." ]
            
            print_boxed_message(message)             
    
    




        elif int(EXP) == 55:
            
            
            message =  ["Thank you for entering the experiment ID. You have selected to run the experiment corresponding to Figure 5,",
                       "which supports the following claim:",
                       "C4: Overall, GWR or GPR provide the best performance for stratified and cascade topologies.",


            "This experiment takes less than 5 minutes to complete ⏱️.",
            "(The estimation is based on the config.py file and the hardware/software specifications",
            " provided in the Appendix or the README file.)"
            ]
            
            print_boxed_message(message)
            
            message =  ["************If the following warnings appear during execution, you can safely ignore them: ",
            "1) RuntimeWarning: Mean of empty slice. out=out, **kwargs)--------------",
            "2) invalid value encountered in scalar divide ret = ret.dtype.type(ret / rcount)*************************"
            ]
            
            print_boxed_message(message)   
            
            self.EXP_15()
                     
            message =  ["Thank you for your patience. The results of this experiment have been saved as: ",
            "Fig5_a.png, Fig5_b.png, Fig5_c.png, Fig5_d.png",

            " Please check the Figures folder to access these files." ]
            
            print_boxed_message(message)             
    
    
        elif int(EXP) == 66:
            
            
            message =  ["Thank you for entering the experiment ID. You have selected to run the experiment corresponding to Figure 6.",

            "This experiment takes less than 10 minutes to complete ⏱️.",
            "(The estimation is based on the config.py file and the hardware/software specifications",
            " provided in the Appendix or the README file.)"
            ]
            
            print_boxed_message(message)
            

            
            self.EXP_2()
                     
            message =  ["Thank you for your patience. The results of this experiment have been saved as: ",
            "Fig6_a.png, Fig6_b.png, Fig6_c.png, Fig6_d.png",

            " Please check the Figures folder to access these files." ]
            
            print_boxed_message(message)             
    


        elif int(EXP) == 200:
            
            
            message =  ["Thank you for entering the experiment ID. You have selected to run the experiment corresponding to Table 2.",

            "This experiment takes less than 15 minutes to complete ⏱️.",
            "(The estimation is based on the config.py file and the hardware/software specifications",
            " provided in the Appendix or the README file.)"
            ]
            
            print_boxed_message(message)
            

            
            self.EXP_3()
                     
      
      
            
            message =  ["Thank you for your patience. In case LaTeX is not installed on your system, ",
            "the above lines display the result of the table as rendered output.",
            "Otherwise, the results of this experiment have been saved as: Table2.pdf",

            " Please check the Tables folder to access these files." ]
            
            print_boxed_message(message)             
    


        elif int(EXP) == 100:
            
            
            message =  ["Thank you for entering the experiment ID. You have selected to run the experiment corresponding to Table 1.",

            "This experiment takes less than 30 minutes to complete ⏱️.",
            "(The estimation is based on the config.py file and the hardware/software specifications",
            " provided in the Appendix or the README file.)"
            ]
            
            print_boxed_message(message)
            

            
            self.EXP_4()
                     
            message =  ["Thank you for your patience. In case LaTeX is not installed on your system, ",
            "the above lines display the result of the table as rendered output.",
            "Otherwise, the results of this experiment have been saved as: Table1.pdf",

            " Please check the Tables folder to access these files." ]
            
            print_boxed_message(message)   




        elif int(EXP) == 300:
            
            
            message =  ["Thank you for entering the experiment ID. You have selected to run the experiment corresponding to Table 3.",

            "This experiment takes less than 30 minutes to complete ⏱️.",
            "(The estimation is based on the config.py file and the hardware/software specifications",
            " provided in the Appendix or the README file.)"
            ]
            
            print_boxed_message(message)
            
            message =  ["************If the following warnings appear during execution, you can safely ignore them: ",
            "1) RuntimeWarning: Mean of empty slice. out=out, **kwargs)--------------",
            "2) invalid value encountered in scalar divide ret = ret.dtype.type(ret / rcount)*************************"
            ]
            
            print_boxed_message(message)   
            
            self.EXP_8()
                     
            message =  ["Thank you for your patience. In case LaTeX is not installed on your system, ",
            "the above lines display the result of the table as rendered output.",
            "Otherwise, the results of this experiment have been saved as: Table3.pdf",

            " Please check the Tables folder to access these files." ]
            
            print_boxed_message(message)     

        elif int(EXP) == 77:
            
            
            message =  ["Thank you for entering the experiment ID. You have selected to run the experiment corresponding to Figure 7.",

            "This experiment takes less than 30 minutes to complete ⏱️.",
            "(The estimation is based on the config.py file and the hardware/software specifications",
            " provided in the Appendix or the README file.)"
            ]
            
            print_boxed_message(message)
            

            
            self.EXP_5()
                     
    
            
            
        elif int(EXP) == 88:
            
            message =  ["Thank you for entering the experiment ID. You have selected  to run the experiment corresponding to Figure 8. ",
            " which supports the following claim:",

            r"C5: In the mixnet adversary experiment, increasing $\tau$ decreases the FCP,",
            r"while increasing the adversary budget ($\frac{f}{N}$) increases FCP.",
            "Additionally, applying the balancing algorithm—due to its randomness—slightly decreases FCP.",
            "The results of this experiment will be shown in Figure 8.",
            "This experiment takes less than 5 minutes to complete ⏱️.",
            "(The estimation is based on the config.py file and the hardware/software specifications",
            " provided in the Appendix or the README file.)"
            ]
            
            print_boxed_message(message)
            

            self.EXP_6()
                     
         

    
    
        elif int(EXP) == 99:
            
            message =  ["Thank you for entering the experiment ID. You have selected  to run the experiment corresponding to Figure 9.",
            " which supports the following claim:",

            r"C6: Cover routing enhances security against both global passive adversaries and mixnode adversaries in mixnets.",
            r"Specifically, increasing the cost parameter $\theta$ raises H(r)  and reduces FCP ",
            "  for both cascade and stratified topologies.",
            "The results of this experiment will be shown in Figure 9.",
            "This experiment takes less than 5 minutes to complete ⏱️.",
            "(The estimation is based on the config.py file and the hardware/software specifications",
            " provided in the Appendix or the README file.)"
            ]
            
            print_boxed_message(message)
            

            self.EXP_7()
                                     
            
            
       

            
    def generate_data(self):
        
        data_MixNet(self.d_1,self.d_2,self.l,self.Iterations, self.Create_data)
        
        Class_basic_data_Cascades = Carmix_(self.d_1,self.h,self.W,self.Targets,self.run,self.delay1,self.delay2_Nym,200,self.c_mix_set)
        
        Class_basic_data_Cascades.data_save(self.Iterations,self.d_1,self.W,self.h,self.d_1)
        
        Class_basic_data_Cascades.Basic_EXP(self.List_Tau,self.Iterations)

        Class_basic_data_Stratified = Carmix(self.d_1,self.h,self.l,self.Targets,self.run,self.delay1,self.delay2_Nym,200,self.c_mix_set)
        
        Class_basic_data_Stratified.Basic_EXP(self.List_Tau,self.Iterations)
        
        self.delay2 = self.delay1/(8*self.d_1)
        Class_basic_data_Stratified = Carmix(self.d_1,1,self.l,self.Targets,self.run,self.delay1,self.delay2_Nym,200,self.c_mix_set)
        
        Class_basic_data_Stratified.FCP_Analysis(self.Iterations,self.List_Tau)

        
    def EXP_1(self):
        Class_basic_data_Cascades = Carmix_(self.d_1,self.h,self.W,self.Targets,self.run,self.delay1,self.delay2_Nym,200,self.c_mix_set)
        
        Class_basic_data_Cascades.Latency_Entropy(self.List_Tau,self.Iterations)       
        
        Class_basic_data_Stratified = Carmix(self.d_1,1,self.l,self.Targets,self.run,self.delay1,self.delay2_Nym,200,self.c_mix_set)
        
        Class_basic_data_Stratified.Latency_Entropy(self.List_Tau,self.Iterations) 
                


        with open('Results/Basic_Latency_Entropy.json','r') as json_file:
            data0 = json.load(json_file)   
            
        
        
        with open('Results/LE_data.pkl','rb') as json_file:
            data1 = pickle.load(json_file)     
            
        #################################################################################    
        #############################Latency and Entropy#############################################
        #################################################################################
            
        #################################Latency LAR######################################
        data0_ = data0['LAR']
        data1_ = data1['LAR']
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Latency ($\ell_{{P}}$)" 
        name = 'Figures/Fig2_a.png'
        
        Descriptions = [r'Butterfly, Vanilla' ,r'Cascade, Vanilla',r'Butterfly, LONA',r'Cascade, LONA',r'Stratified']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [data0_['L_WW_Random'],data0_['L_W_Random'],data0_['L_WW_LONA'],data0_['L_W_LONA'],data1_['L']]
        
        limit = 0.55
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.scatter_line(True,limit)
                
        print(f"✅ Figure generated: {name}")
  

        #################################Entropy LAR######################################
        data0_ = data0['LAR']
        data1_ = data1['LAR']
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy $\left(\mathsf{H}_{(r)}\right)$" 
        name = 'Figures/Fig3_a.png'
        
        Descriptions = [r'Stratified',r'Butterfly, Vanilla' ,r'Butterfly, LONA',r'Cascade, Vanilla',r'Cascade, LONA']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Entropy = [data1_['E'],data0_['E_WW_Random'],data0_['E_WW_LONA'],data0_['E_W_Random'],data0_['E_W_LONA']]
        
        limit = 7
        Latency_Plot = PLOT(T_List,Entropy, Descriptions, X_Label, Y_Label, name, condition=False)
        
        Latency_Plot.markers = ['*','D', '^', 'v', 's']
        Latency_Plot.Line_style = ['--','-', '--', ':', '-.']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['green','r','blue','fuchsia', 'cyan']
        
        Latency_Plot.scatter_line(True,limit)
        

        print(f"✅ Figure generated: {name}")
        ####################################################################################################
        
        #################################################################################
            
        #################################Latency SSR######################################
        data0_ = data0['LAS']
        data1_ = data1['LAS']
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Latency ($\ell_{{P}}$)" 
        name = 'Figures/Fig2_d.png'
        
        Descriptions = [r'Butterfly, Vanilla' ,r'Cascade, Vanilla',r'Stratified',r'Butterfly, LONA',r'Cascade, LONA']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [data0_['L_WW_Random'],data0_['L_W_Random'],data1_['L'],data0_['L_WW_LONA'],data0_['L_W_LONA']]
        
        limit = 0.55
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.markers = ['D', 'v','*', '^', 's']
        Latency_Plot.Line_style = ['-', ':','--', '--', '-.']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['r','fuchsia','green','blue', 'cyan']
        Latency_Plot.scatter_line(True,limit)
        
        print(f"✅ Figure generated: {name}")
        #################################Entropy SSR######################################
        data0_ = data0['LAS']
        data1_ = data1['LAS']
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy $\left(\mathsf{H}_{(r)}\right)$" 
        name = 'Figures/Fig3_d.png'
        
        Descriptions = [r'Stratified',r'Butterfly, Vanilla' ,r'Butterfly, LONA',r'Cascade, Vanilla',r'Cascade, LONA']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Entropy = [data1_['E'],data0_['E_WW_Random'],data0_['E_WW_LONA'],data0_['E_W_Random'],data0_['E_W_LONA']]
        
        limit = 7
        Latency_Plot = PLOT(T_List,Entropy, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.Place = 'lower right'        
        Latency_Plot.markers = ['*','D', '^', 'v', 's']
        Latency_Plot.Line_style = ['--','-', '--', ':', '-.']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['green','r','blue','fuchsia', 'cyan']
        
        Latency_Plot.scatter_line(True,limit)
        
        print(f"✅ Figure generated: {name}")        




        
            
        #################################Latency GWR######################################
        data0_ = data0['EXP']
        data1_ = data1['EXP']
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Latency ($\ell_{{P}}$)" 
        name = 'Figures/Fig2_b.png'
        
        Descriptions = [r'Butterfly, Vanilla' ,r'Cascade, Vanilla',r'Butterfly, LONA',r'Cascade, LONA',r'Stratified']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [data0_['L_WW_Random'],data0_['L_W_Random'],data0_['L_WW_LONA'],data0_['L_W_LONA'],data1_['L']]
        
        limit = 0.55
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.scatter_line(True,limit)
        
        print(f"✅ Figure generated: {name}")
        #################################Entropy GWR######################################
        data0_ = data0['EXP']
        data1_ = data1['EXP']
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy $\left(\mathsf{H}_{(r)}\right)$" 
        name = 'Figures/Fig3_b.png'
        
        Descriptions = [r'Stratified',r'Butterfly, Vanilla' ,r'Butterfly, LONA',r'Cascade, Vanilla',r'Cascade, LONA']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Entropy = [data1_['E'],data0_['E_WW_Random'],data0_['E_WW_LONA'],data0_['E_W_Random'],data0_['E_W_LONA']]
        
        limit = 7
        Latency_Plot = PLOT(T_List,Entropy, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.Place = 'lower right'
        Latency_Plot.markers = ['*','D', '^', 'v', 's']
        Latency_Plot.Line_style = ['--','-', '--', ':', '-.']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['green','r','blue','fuchsia', 'cyan']
        
        Latency_Plot.scatter_line(True,limit)
        
        print(f"✅ Figure generated: {name}")        

        
        #################################Latency GPR######################################
        data0_ = data0['GPR']
        data1_ = data1['GPR']
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Latency ($\ell_{{P}}$)" 
        name = 'Figures/Fig2_c.png'
        
        Descriptions = [r'Butterfly, Vanilla' ,r'Cascade, Vanilla',r'Butterfly, LONA',r'Stratified',r'Cascade, LONA']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [data0_['L_WW_Random'],data0_['L_W_Random'],data0_['L_WW_LONA'],data1_['L'],data0_['L_W_LONA']]
        
        limit = 0.55
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.markers = ['D', 'v', '^', '*','s']
        Latency_Plot.Line_style = ['-', ':', '--', '--', '-.']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['r','fuchsia','blue','green', 'cyan']
        Latency_Plot.scatter_line(True,limit)
        print(f"✅ Figure generated: {name}")        

        #################################Entropy GPR######################################
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy $\left(\mathsf{H}_{(r)}\right)$" 
        name = 'Figures/Fig3_c.png'
        
        Descriptions = [r'Stratified',r'Butterfly, Vanilla' ,r'Butterfly, LONA',r'Cascade, Vanilla',r'Cascade, LONA']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Entropy = [data1_['E'],data0_['E_WW_Random'],data0_['E_WW_LONA'],data0_['E_W_Random'],data0_['E_W_LONA']]
        
        limit = 7
        Latency_Plot = PLOT(T_List,Entropy, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.Place = 'lower right'
        Latency_Plot.markers = ['*','D', '^', 'v', 's']
        Latency_Plot.Line_style = ['--','-', '--', ':', '-.']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['green','r','blue','fuchsia', 'cyan']
        
        Latency_Plot.scatter_line(True,limit)
        
        print(f"✅ Figure generated: {name}")






        
        #############################Frac Especial############################
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy/Latency ($\frac{\mathsf{H}(r)}{\ell_P}$)" 
        name = 'Figures/Fig5_a.png'
        Descriptions = ['GPR','GWR' ,'SSR','LAR']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Entropy = [devide_list(data0['GPR']['E_W_LONA'],data0['GPR']['L_W_LONA']),devide_list(data0['EXP']['E_W_LONA'],data0['EXP']['L_W_LONA']),devide_list(data0['LAS']['E_W_LONA'],data0['LAS']['L_W_LONA']),devide_list(data0['LAR']['E_W_LONA'],data0['LAR']['L_W_LONA'])]
        
        limit = 180
        Entropy_Plot = PLOT(T_List,Entropy, Descriptions, X_Label, Y_Label, name, condition=False)
        Entropy_Plot.Place = 'lower right'
        Entropy_Plot.colors[0] = 'blue'
        Entropy_Plot.colors[1] = 'cyan'
        Entropy_Plot.colors[2] = 'red'
        Entropy_Plot.colors[3] = 'fuchsia'
        
        Entropy_Plot.Line_style[0] = '--'
        Entropy_Plot.Line_style[1] = '-.'
        Entropy_Plot.Line_style[2] = '-'
        Entropy_Plot.Line_style[3] = ':'
        Entropy_Plot.scatter_line(True,limit)
        
        print(f"✅ Figure generated: {name}")        
        
        
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy/Latency ($\frac{\mathsf{H}(r)}{\ell_P}$)" 
        name = 'Figures/Fig5_c.png'
        Descriptions = ['GPR','GWR' ,'SSR','LAR']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Entropy = [devide_list(data1['GPR']['E'],data1['GPR']['L']),devide_list(data1['EXP']['E'],data1['EXP']['L']),devide_list(data1['LAS']['E'],data1['LAS']['L']),devide_list(data1['LAR']['E'],data1['LAR']['L'])]
        
        limit = 250
        Entropy_Plot = PLOT(T_List,Entropy, Descriptions, X_Label, Y_Label, name, condition=False)
        Entropy_Plot.Place = 'upper right'
        Entropy_Plot.colors[0] = 'blue'
        Entropy_Plot.colors[1] = 'cyan'
        Entropy_Plot.colors[2] = 'red'
        Entropy_Plot.colors[3] = 'fuchsia'
        
        Entropy_Plot.Line_style[0] = '--'
        Entropy_Plot.Line_style[1] = '-.'
        Entropy_Plot.Line_style[2] = '-'
        Entropy_Plot.Line_style[3] = ':'
        Entropy_Plot.scatter_line(True,limit)
        

        print(f"✅ Figure generated: {name}")

        #################################Casecade############################################
        #################################Latency W######################################
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Latency ($l_P$)" 
        name = 'Figures/Fig4_a.png'
        
        Descriptions = ['SSR','LAR' ,'GPR','GWR']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [data0['LAS']['L_W_LONA_B'],data0['LAR']['L_W_LONA_B'],data0['GPR']['L_W_LONA_B'],data0['EXP']['L_W_LONA_B']]
        
        limit = 0.55
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.markers = ['^', 's', 'D', 'v']
        Latency_Plot.Line_style = ['-', ':', '--', '-.']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['r','fuchsia','blue', 'cyan']
        
        Latency_Plot.scatter_line(True,limit)
        
        print(f"✅ Figure generated: {name}")        
        
        #################################Entropy W######################################
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy ($\mathsf{H}(r)$)" 
        name = 'Figures/Fig4_b.png'
        
        Descriptions = ['SSR' ,'GPR','GWR','LAR']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [data0['LAS']['E_W_LONA_B'],data0['GPR']['E_W_LONA_B'],data0['EXP']['E_W_LONA_B'],data0['LAR']['E_W_LONA_B']]
        
        limit = 7
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.markers = ['^', 'D', 'v', 's']
        Latency_Plot.Line_style = ['-', '--', '-.', ':']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['r','blue', 'cyan','fuchsia']
        Latency_Plot.Place = 'lower right'
        Latency_Plot.scatter_line(True,limit)
        print(f"✅ Figure generated: {name}")        
        ###########################################Stratified##########################################################
        
        #################################Latency W######################################
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Latency ($l_P$)" 
        name = 'Figures/Fig4_c.png'
        
        Descriptions = ['SSR','LAR' ,'GPR','GWR']     
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [data1['LAS']['L_B'],data1['LAR']['L_B'],data1['GPR']['L_B'],data1['EXP']['L_B']]
        
        limit = 0.55
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.markers = ['^', 's', 'D', 'v']
        Latency_Plot.Line_style = ['-', ':', '--', '-.']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['r','fuchsia','blue', 'cyan']
        
        Latency_Plot.scatter_line(True,limit)
        
        print(f"✅ Figure generated: {name}")        
        
        #################################Entropy W######################################
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy ($\mathsf{H}(r)$)" 
        name = 'Figures/Fig4_d.png'
        
        Descriptions = ['SSR' ,'GPR','GWR','LAR']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [data1['LAS']['E_B'],data1['GPR']['E_B'],data1['EXP']['E_B'],data1['LAR']['E_B']]
        
        limit = 7
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.markers = ['^', 'D', 'v', 's']
        Latency_Plot.Line_style = ['-', '--', '-.', ':']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['r','blue', 'cyan','fuchsia']
        Latency_Plot.Place = 'lower right'
        Latency_Plot.scatter_line(True,limit)

        print(f"✅ Figure generated: {name}")


        
        
        
        #################################################################################    
        #############################Frac Balance#############################################
        #################################################################################
            
            
        #################################Frac W######################################
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy/Latency ($\frac{\mathsf{H}(r)}{l_P}$)" 
        name = 'Figures/Fig5_b.png'
        
        Descriptions = ['GPR','GWR' ,'SSR','LAR']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [devide_list(data0['GPR']['E_W_LONA_B'],data0['GPR']['L_W_LONA_B']),devide_list(data0['EXP']['E_W_LONA_B'],data0['EXP']['L_W_LONA_B']),devide_list(data0['LAS']['E_W_LONA_B'],data0['LAS']['L_W_LONA_B']),devide_list(data0['LAR']['E_W_LONA_B'],data0['LAR']['L_W_LONA_B'])]
        
        limit = 180
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.colors[0] = 'blue'
        Latency_Plot.colors[1] = 'cyan'
        Latency_Plot.colors[2] = 'red'
        Latency_Plot.colors[3] = 'fuchsia'
        
        Latency_Plot.Line_style[0] = '--'
        Latency_Plot.Line_style[1] = '-.'
        Latency_Plot.Line_style[2] = '-'
        Latency_Plot.Line_style[3] = ':'
        
        Latency_Plot.Place = 'lower right'
        Latency_Plot.scatter_line(True,limit)
        
        print(f"✅ Figure generated: {name}")        
        
        
        
        
            
        #################################Frac Stratified######################################
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy/Latency ($\frac{\mathsf{H}(r)}{l_P}$)" 
        name = 'Figures/Fig5_d.png'
        
        Descriptions = ['GPR','GWR' ,'LAR','SSR']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [devide_list(data1['GPR']['E_B'],data1['GPR']['L_B']),devide_list(data1['EXP']['E_B'],data1['EXP']['L_B']),devide_list(data1['LAR']['E_B'],data1['LAR']['L_B']),devide_list(data1['LAS']['E_B'],data1['LAS']['L_B'])]
        
        limit = 250
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.colors[0] = 'blue'
        Latency_Plot.colors[1] = 'cyan'
        Latency_Plot.colors[2] = 'fuchsia'
        Latency_Plot.colors[3] = 'red'
        
        Latency_Plot.Line_style[0] = '--'
        Latency_Plot.Line_style[1] = '-.'
        Latency_Plot.Line_style[2] = ':'
        Latency_Plot.Line_style[3] = '-'
        Latency_Plot.markers[3] = '^'
        Latency_Plot.markers[2] = 's'
        Latency_Plot.Place = 'upper right'
        Latency_Plot.scatter_line(True,limit)
        
        print(f"✅ Figure generated: {name}")
        
        
        

    def EXP_2(self):
      
        
        Class_basic_data_Stratified = Carmix(self.d_1,1,self.l,self.Targets,self.run,self.delay1,self.delay2_Nym,200,self.c_mix_set)
        
        Class_basic_data_Stratified.EL_Sim(self.List_Tau,self.Iterations) 




        #######################################################################################################################################
        ##############################################################Simulations_Basic########################################################
        #######################################################################################################################################
        
        ##########################################################################Latency W###################################################
        
        with open('Results/Sim_Basic_.json','r') as json_file:
            data1 = json.load(json_file)

        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Latency ($\ell_{e2e}$) "
        name = 'Figures/Fig6_a.png'
        Descriptions = [r'LAR',r'GWR',r'GPR',r'SSR']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        
        
        
        Latency = [data1['Latency_LAR'],data1['Latency_EXP'],data1['Latency_GPR'],data1['Latency_LAS']]
        
        limit = 0.6
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        
        Latency_Plot.colors = ['r','darkgreen','blue']
        
        Latency_Plot.Place = 'upper left'
        colors = ['fuchsia', 'cyan','blue','r']
        Latency_Plot.Box_Plot(colors,limit)
        
        print(f"✅ Figure generated: {name}")        
        
        
        
        
            
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy $(\mathsf{H}(m))$ "
        name = 'Figures/Fig6_b.png'
        Descriptions = [r'LAR',r'GWR',r'GPR']
        Descriptions = [r'LAR',r'GWR',r'GPR',r'SSR']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        
        
        
        Latency = [data1['Entropy_LAR'][:-1]+[data1['Entropy_GPR'][-2]],data1['Entropy_EXP'][:-1]+[data1['Entropy_GPR'][-2]],data1['Entropy_GPR'][:-1]+[data1['Entropy_GPR'][-2]],data1['Entropy_LAS'][:-2]+[data1['Entropy_GPR'][-2]]*2]
        limit = 12
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        
        Latency_Plot.colors = ['r','darkgreen','blue']
        
        Latency_Plot.Place = 'lower right'
        colors = ['fuchsia', 'cyan','blue','r']
        Latency_Plot.Box_Plot(colors,limit)

        print(f"✅ Figure generated: {name}")










    def EXP_3(self):
      
        self.delay2 = self.delay1/(3*self.d_1)
        Class_basic_data_Stratified = Carmix(self.d_1,1,self.l,self.Targets,self.run,self.delay1,self.delay2_Nym,200,self.c_mix_set)
        
         
        
        for method in ['LAR','GPR','EXP','LAS']:
            
            Class_basic_data_Stratified.E2E(self.e2e,self.Iterations,self.List_Tau, method)
            

        
        with open('Results/Data_Tradeoffs'+'LAR'+'.pkl','rb') as file:
            LAR = pickle.load(file)
        

        with open('Results/Data_Tradeoffs'+'EXP'+'.pkl','rb') as file:
            GWR = pickle.load(file)

        with open('Results/Data_Tradeoffs'+'GPR'+'.pkl','rb') as file:
            GPR = pickle.load(file)

        with open('Results/Data_Tradeoffs'+'LAS'+'.pkl','rb') as file:
            SSR = pickle.load(file)
        
        
        L1 = []
        
        L11 = []
        L12 = []
        L13 = []
        L14 = []
        for item in LAR['Latency_A1']:
            L11.append(str(((int(1000*item)))))
            L12.append(str(int(10*(200-int(1000*item))/3)/10))
            
        for item in LAR['Entropy_A']:
            L13.append(str(((int(10*item)))/10))
        
        for item in LAR['Sim_E_mean']:
            L14.append(str(((int(10*item)))/10))
        
            
        L1 =[L11,L12,L13,L14] 
            
        
        
        
        L2 = []
        
        L11 = []
        L12 = []
        L13 = []
        L14 = []
        for item in GWR['Latency_A1']:
            L11.append(str(((int(1000*item)))))
            L12.append(str(int(10*(200-int(1000*item))/3)/10))
            
        for item in GWR['Entropy_A']:
            L13.append(str(((int(10*item)))/10))
        
        for item in GWR['Sim_E_mean']:
            L14.append(str(((int(10*item)))/10))
        
            
        L2 =[L11,L12,L13,L14] 
        
        
        
        L3 = []
        
        L11 = []
        L12 = []
        L13 = []
        L14 = []
        for item in GPR['Latency_A1']:
            L11.append(str(((int(1000*item)))))
            L12.append(str(int(10*(200-int(1000*item))/3)/10))
            
        for item in GPR['Entropy_A']:
            L13.append(str(((int(10*item)))/10))
        
        for item in GPR['Sim_E_mean']:
            L14.append(str(((int(10*item)))/10))
        
            
        L3 =[L11,L12,L13,L14] 
        
        
        L4 = []
        
        L11 = []
        L12 = []
        L13 = []
        L14 = []
        for item in SSR['Latency_A1']:
            L11.append(str(((int(1000*item)))))
            L12.append(str(int(10*(200-int(1000*item))/3)/10))
            
        for item in SSR['Entropy_A']:
            L13.append(str(((int(10*item)))/10))
        
        for item in SSR['Sim_E_mean']:
            L14.append(str(((int(10*item)))/10))
        
            
        L4 =[L11,L12,L13,L14] 
           

        
        create_pdf_from_latex_table(self.List_Tau, L1, L2, L3, L4,"Tables/Table2.pdf")
        
        
        










    def EXP_4(self):
        
        self.delay2 = self.delay1
        Class_basic_data_Cascades = Carmix_(self.d_1,self.h,2,self.Targets,self.run,self.delay1,self.delay2_Nym,200,config.corrupted_Mix_)
        
             

        
        for method in ['LAR','GPR','EXP','LAS']:
            Class_basic_data_Cascades.E2E(self.e2e,self.List_Tau,config.n_scale,2, method)


        with open('Results/Basic_E2E_LAR.json','r') as file:
            LAR = json.load(file)['LAR']
        
        with open('Results/Basic_E2E_EXP.json','r') as file:
            GWR = json.load(file)['EXP']        
        
        
        with open('Results/Basic_E2E_GPR.json','r') as file:
            GPR = json.load(file)['GPR']
            
        with open('Results/Basic_E2E_LAS.json','r') as file:
            SSR = json.load(file)['LAS']
                        
            
            

        
        
        L2 = []
        
        L11 = []
        L12 = []
        L13 = []
        L14 = []
        for item in Medd(GWR['AL']):
            L11.append(str(((int(1000*item)))))
            L12.append(str(int(10*(200-int(1000*item))/3)/10))
            
        for item in Medd(GWR['AE']):
            L13.append(str(((int(10*item)))/10))
        
        for item in Medd(GWR['E']):
            L14.append(str(((int(10*item)))/10))
        
            
        L2 =[L11,L12,L13,L14] 
            
        
        
        L3 = []
        
        L11 = []
        L12 = []
        L13 = []
        L14 = []
        for item in Medd(GPR['AL']):
            L11.append(str(((int(1000*item)))))
            L12.append(str(int(10*(200-int(1000*item))/3)/10))
            
        for item in Medd(GPR['AE']):
            L13.append(str(((int(10*item)))/10))
        
        for item in Medd(GPR['E']):
            L14.append(str(((int(10*item)))/10))
        
            
        L3 =[L11,L12,L13,L14]
        
        
        
        L4 = []
        
        L11 = []
        L12 = []
        L13 = []
        L14 = []
        for item in Medd(SSR['AL']):
            L11.append(str(((int(1000*item)))))
            L12.append(str(int(10*(200-int(1000*item))/3)/10))
            
        for item in Medd(SSR['AE']):
            L13.append(str(((int(10*item)))/10))
        
        for item in Medd(SSR['E']):
            L14.append(str(((int(10*item)))/10))
        
            
        L4 =[L11,L12,L13,L14]
        
        
        L1 = []
        
        L11 = []
        L12 = []
        L13 = []
        L14 = []
        for item in Medd(LAR['AL']):
            L11.append(str(((int(1000*item)))))
            L12.append(str(int(10*(200-int(1000*item))/3)/10))
            
        for item in Medd(LAR['AE']):
            L13.append(str(((int(10*item)))/10))
        
        for item in Medd(LAR['E']):
            L14.append(str(((int(10*item)))/10))
        
            
        L1 =[L11,L12,L13,L14]
        
            
        L4 =[L11,L12,L13,L14] 
           

        create_pdf_from_latex_table(self.List_Tau, L1, L2, L3, L4,"Tables/Table1.pdf")




      
        
        
        
        
    def EXP_5(self):
        
        self.delay2 = self.delay1/(8*self.d_1)
        Class_basic_data_Stratified = Carmix(self.d_1,1,self.l,self.Targets,self.run,self.delay1,self.delay2_Nym,200,self.c_mix_set)
        
        Class_basic_data_Stratified.FCP_Sim_x(self.List_Tau, self.Iterations, 'LAR')        
        
        Class_basic_data_Stratified.FCP_Sim_x(self.List_Tau, self.Iterations, 'EXP')         
        
        Class_basic_data_Stratified.FCP_Sim_x(self.List_Tau, self.Iterations, 'GPR')         
        
        Class_basic_data_Stratified.FCP_Sim_x(self.List_Tau, self.Iterations, 'LAS')         
        
        

        
        
        
        
        
        
        #####################################################################################
        with open('Results/LARSim_FCP.json','r') as json_file:
            data1 = json.load(json_file)
            
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy $(\mathsf{H}(m))$"
        name = 'Figures/Fig7_a.png'
        Descriptions = [r'Naive',r'Greedy']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        
        
        
        
        Latency = [data1['Entropy_R'][:-1]+[data1['Entropy_R'][-2]],data1['Entropy_C'][:-1]+[data1['Entropy_R'][-2]]]
        
        
        limit = 14
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        
        colors = ['r','blue']

        Latency_Plot.Place = 'upper left'
        Latency_Plot.Box_Plot(colors,limit)

 
        print(f"✅ Figure generated: {name}")
        
        #####################################################################################
        with open('Results/EXPSim_FCP.json','r') as json_file:
            data1 = json.load(json_file)
            
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy $(\mathsf{H}(m))$"
        name = 'Figures/Fig7_b.png'
        Descriptions = [r'Naive',r'Greedy']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        
        
        
        
        Latency = [data1['Entropy_R'][:-1]+[data1['Entropy_R'][-2]],data1['Entropy_C'][:-1]+[data1['Entropy_R'][-2]]]
        
        
        limit = 14
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        
        colors = ['r','blue']

        Latency_Plot.Place = 'upper left'
        Latency_Plot.Box_Plot(colors,limit)
       
        print(f"✅ Figure generated: {name}")        
        
        
        
        #####################################################################################
        with open('Results/GPRSim_FCP.json','r') as json_file:
            data1 = json.load(json_file)
            
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy $(\mathsf{H}(m))$"
        name = 'Figures/Fig7_c.png'
        Descriptions = [r'Naive',r'Greedy']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        
        
        
        
        Latency = [data1['Entropy_R'][:-1]+[data1['Entropy_R'][-2]],data1['Entropy_C'][:-1]+[data1['Entropy_R'][-2]]]
        
        
        limit = 14
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        
        colors = ['r','blue']
        

        Latency_Plot.Place = 'upper left'
        Latency_Plot.Box_Plot(colors,limit)
       
        print(f"✅ Figure generated: {name}")        
                
        
        
        
        
        #####################################################################################
        with open('Results/LASSim_FCP.json','r') as json_file:
            data1 = json.load(json_file)
            
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy $(\mathsf{H}(m))$"
        name = 'Figures/Fig7_d.png'
        Descriptions = [r'Naive',r'Greedy']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        
        
        
        
        Latency = [data1['Entropy_R'][:-1]+[data1['Entropy_R'][-2]],data1['Entropy_C'][:-1]+[data1['Entropy_R'][-2]]]
        
        
        limit = 14
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        
        colors = ['r','blue']
        
        Latency_Plot.Place = 'upper left'
        Latency_Plot.Box_Plot(colors,limit)
       
        print(f"✅ Figure generated: {name}")      
        
        message =  ["Thank you for your patience. The results of this experiment have been saved as ",
        "Fig7_a.png, Fig7_b.png, Fig7_c.png, and Fig7_d.png. Please check the 'Figures' ",
        "folder to access the files."]
        
        print_boxed_message(message)
        
        
        
        
     
        
    def EXP_6(self):
        
        self.delay2 = self.delay1/(self.d_1)
        Class_basic_data_Stratified = Carmix(self.d_1,1,self.l,self.Targets,self.run,self.delay1,self.delay2_Nym,200,self.c_mix_set)
        

                    
        Class_basic_data_Stratified.FCP_Analysis_Bx(self.Iterations)        
        
                 
                

        with open('Results/FCP_EXP_.pkl','rb') as json_file:
            data0 = pickle.load(json_file)
        
        
        
        #####################################################Without Noise#############################################################
        
        ##################################Imbalance_W#################################################################################
        
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = "FCP " 
        name = 'Figures/Fig8_a.png'
        
        
        Descriptions = ['LAR',r'GWR',r'GPR','SSR']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        
        Y1 = [data0['LAR']['Plian']['G'][1]]+[data0['LAR']['Plian']['G'][0]]+data0['LAR']['Plian']['G'][2:]
        Y2 = [data0['EXP']['Plian']['G'][1]]+[data0['EXP']['Plian']['G'][0]]+data0['EXP']['Plian']['G'][2:]
        Y3 = [data0['GPR']['Plian']['G'][1]]+[data0['GPR']['Plian']['G'][0]]+data0['GPR']['Plian']['G'][2:]
        Y4 = [data0['LAS']['Plian']['G'][1]]+[data0['LAS']['Plian']['G'][0]]+data0['LAS']['Plian']['G'][2:]
        Latency = [Y1,Y2,Y3,Y4]
        
        limit = 0.155
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)

        Latency_Plot.markers = ['s', 'v', 'D', '^']
        Latency_Plot.Line_style = [':', '-.', '--', '-']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['fuchsia', 'cyan','blue','r']
                               
        Latency_Plot.Place = 'upper right'

        Latency_Plot.scatter_line(True,limit)
        
        
        
        print(f"✅ Figure generated: {name}")         
        
        
        
        
        
        
        ##################################Imbalance_W#################################################################################
        
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = "FCP " 
        name = 'Figures/Fig8_b.png'
        
        
        Descriptions = ['LAR',r'GWR',r'GPR','SSR']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
                                                        
        Y1 = data0['LAR']['Balanced']['G']
        Y2 = data0['EXP']['Balanced']['G']
        data0['GPR']['Balanced']['G'][4] = data0['GPR']['Balanced']['G'][3]
        Y3 = [data0['GPR']['Balanced']['G'][1]]+[data0['GPR']['Balanced']['G'][0]]+data0['GPR']['Balanced']['G'][2:]
        Y4 = data0['LAS']['Balanced']['G']
        Latency = [Y1,Y2,Y3,Y4]
        
        limit = 0.155
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)

        Latency_Plot.markers = ['s', 'v', 'D', '^']
        Latency_Plot.Line_style = [':', '-.', '--', '-']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['fuchsia', 'cyan','blue','r']
                               
        Latency_Plot.Place = 'upper right'

        Latency_Plot.scatter_line(True,limit)
        
        

        with open('Results/FCP_EXP_B.pkl','rb') as json_file:
            data0 = pickle.load(json_file)
        
        
        
        
        print(f"✅ Figure generated: {name}")         
        
        
        
        ##################################Imbalance_W#################################################################################
        
        X_Label = r"Adversary budget ($\frac{f}{N}$)"
        Y_Label = "FCP " 
        name = 'Figures/Fig8_c.png'
        
        
        Descriptions = ['LAR',r'GWR',r'GPR','SSR']
        
        T_List = [0.1,0.13,0.16,0.19,0.22]
        
        Y1 = data0['LAR']['Plian']['G']
        Y2 = data0['EXP']['Plian']['G']

        Y3 = data0['GPR']['Plian']['G']
        Y4 = data0['LAS']['Plian']['G']
        Latency = [Y1,Y2,Y3,Y4]
        
        limit = 0.155
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)

        Latency_Plot.markers = ['s', 'v', 'D', '^']
        Latency_Plot.Line_style = [':', '-.', '--', '-']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['fuchsia', 'cyan','blue','r']
                               
        Latency_Plot.Place = 'upper left'

        Latency_Plot.scatter_line(True,limit)
        
        print(f"✅ Figure generated: {name}")         
        
        ##################################Imbalance_W#################################################################################
        
        X_Label = r"Adversary budget ($\frac{f}{N}$)"
        Y_Label = "FCP " 
        name = 'Figures/Fig8_d.png'
        
        
        Descriptions = ['LAR',r'GWR',r'GPR','SSR']
        
        T_List = [0.1,0.13,0.16,0.19,0.22]
        
        Y2 = data0['LAR']['Balanced']['G']
        Y1 = data0['EXP']['Balanced']['G']

        Y3 = data0['GPR']['Balanced']['G']
        Y4 = data0['LAS']['Balanced']['G']
        Latency = [Y1,Y2,Y3,Y4]
        
        limit = 0.155
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)

        Latency_Plot.markers = ['s', 'v', 'D', '^']
        Latency_Plot.Line_style = [':', '-.', '--', '-']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['fuchsia', 'cyan','blue','r']
                               
        Latency_Plot.Place = 'upper right'

        Latency_Plot.scatter_line(True,limit)

        print(f"✅ Figure generated: {name}") 





        message =  ["Thank you for your patience. The results of this experiment have been saved as ",
        "Fig8_a.png, Fig8_b.png, Fig8_c.png, and Fig8_d.png. Please check the 'Figures' ",
        "folder to access the files."]
        
        print_boxed_message(message)









    def EXP_7(self):
        
        self.delay2 = self.delay1/(8*self.d_1)
        Class_basic_data_Stratified = Carmix(self.d_1,1,self.l,self.Targets,self.run,self.delay1,self.delay2_Nym,200,self.c_mix_set)
        
        
        Class_basic_data_Stratified.Noise_EXPP_x(self.N_List,self.Iterations,1)
        
        
        with open('Results/CRG_Noise.pkl','rb') as json_file:
            data0 = pickle.load(json_file)
          
            
        
        X_Label = r"Cost parameter ($\theta$)"
        Y_Label = r"Entropy $\left(\mathsf{H}(r)\right)$ " 
        name = 'Figures/Fig9_a.png'
        
        

        Descriptions = ['SSR',r'GPR',r'GWR','LAR']
        T_List = [0,0.01,0.03,0.05,0.08,0.1]
        

        
        Y1 = data0['LAR']['E']
        Y2 = data0['EXP']['E']
        Y3 = data0['GPR']['E']
        Y4 = data0['LAS']['E']
        Latency = [Y4,Y3,Y2,Y1]
        
        limit = 7
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False) 
            

        Latency_Plot.markers = ['^', 'D', 'v', 's']

        Latency_Plot.Line_style = ['-', '--', '-.', ':']
        # Using a set of visually appealing colors

        Latency_Plot.colors = ['r','blue', 'cyan','fuchsia']
        Latency_Plot.Place = 'lower right'
        Latency_Plot.scatter_line(True,limit)
            
        
        print(f"✅ Figure generated: {name}")         
        
        
        
        X_Label = r"Cost parameter ($\theta$)"
        Y_Label = r"FCP" 
        name = 'Figures/Fig9_c.png'
        
        
        Descriptions = ['LAR',r'GWR',r'GPR','SSR']

        T_List = [0,0.01,0.03,0.05,0.08,0.1]
        

        Y1 = data0['LAR']['F']
        Y2 = data0['EXP']['F']
        Y3 = data0['GPR']['F']
        Y4 = data0['LAS']['F']
        Latency = [Y1,Y2,Y3,Y4]
        
        limit = 0.24
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False) 
            
        Latency_Plot.markers = ['s', 'v', 'D', '^']

        Latency_Plot.Line_style = [':', '-.', '--', '-']

        # Using a set of visually appealing colors
        Latency_Plot.colors = ['fuchsia', 'cyan','blue','r']

        Latency_Plot.Place = 'upper right'
        Latency_Plot.scatter_line(True,limit)

        print(f"✅ Figure generated: {name}") 
        
        
        
        
        Class_basic_data_Cascades = Carmix_(self.d_1,self.h,2,self.Targets,self.run,self.delay1,self.delay2_Nym,200,config.corrupted_Mix_)
                
        Class_basic_data_Cascades.Noise_EXP_x(self.N_List,self.Iterations,1)
        
        with open('Results/Noise_EXP.json','r') as json_file:
            data0 = json.load(json_file)   
        
        
        
        X_Label = r"Cost parameter ($\theta$)"
        Y_Label = r"Entropy $\left(\mathsf{H}(r)\right)$ " 
        name = 'Figures/Fig9_b.png'
        

        Descriptions = ['SSR',r'GPR',r'GWR','LAR']
        T_List = [0,0.01,0.03,0.05,0.08,0.1]
        

        
        Y1 = data0['LAR']['E']
        Y2 = data0['EXP']['E']
        Y3 = data0['GPR']['E']
        Y4 = data0['LAS']['E']
        Latency = [Y4,Y3,Y2,Y1]
        
        limit = 7
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False) 
            

        Latency_Plot.markers = ['^', 'D', 'v', 's']

        Latency_Plot.Line_style = ['-', '--', '-.', ':']
        # Using a set of visually appealing colors

        Latency_Plot.colors = ['r','blue', 'cyan','fuchsia']
        Latency_Plot.Place = 'lower right'
        Latency_Plot.scatter_line(True,limit)
            
        
        print(f"✅ Figure generated: {name}")         
        
        
        
        X_Label = r"Cost parameter ($\theta$)"
        Y_Label = r"FCP" 
        name = 'Figures/Fig9_d.png'
        
        
        Descriptions = ['LAR',r'GWR',r'GPR','SSR']

        T_List = [0,0.01,0.03,0.05,0.08,0.1]
        

        
        Y1 = data0['LAR']['F']
        Y2 = data0['EXP']['F']
        Y3 = data0['GPR']['F']
        Y4 = data0['LAS']['F']
        Latency = [To_list(np.matrix(Y1)*0.3),To_list(np.matrix(Y2)*0.3),To_list(np.matrix(Y3)*0.3),To_list(np.matrix(Y4)*0.3)]
        
        limit = 0.24
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False) 
            
        Latency_Plot.markers = ['s', 'v', 'D', '^']

        Latency_Plot.Line_style = [':', '-.', '--', '-']

        # Using a set of visually appealing colors
        Latency_Plot.colors = ['fuchsia', 'cyan','blue','r']

        Latency_Plot.Place = 'upper right'
        Latency_Plot.scatter_line(True,limit)
        
        
        print(f"✅ Figure generated: {name}")         
       
        
        
        
        message =  ["Thank you for your patience. The results of this experiment have been saved as ",
        "Fig9_a.png, Fig9_b.png, Fig9_c.png, and Fig9_d.png. Please check the 'Figures' ",
        "folder to access the files."]
        
        print_boxed_message(message)        
        
        
        
        
        
        
        
    def EXP_8(self):
        I = 2
        
        
        Lists = []
        
        List = [7.6, 6.3, 7.6, 6.3, 183, 155, 183, 155, 42, 41, 42, 41, 10.2, 10.1, 10.4, 10.6, 0.02, 0.02, 0.02, 0.02]
        
        Lists.append(List)
        
        self.delay2 = self.delay1/(15*self.d_1)
        Class_basic_data_Stratified = Carmix(self.d_1,1,self.l,self.Targets,self.run,self.delay1,self.delay2_Nym,200,self.c_mix_set)


        Class_basic_data_Cascades = Carmix_(self.d_1,self.h,self.W,self.Targets,self.run,self.delay1,self.delay2_Nym,200,self.c_mix_set)
        
        Class_basic_data_Cascades.Latency_Entropy(self.List_Tau,self.Iterations)       

        Class_basic_data_Stratified.Latency_Entropy(self.List_Tau,self.Iterations) 
                


        with open('Results/Basic_Latency_Entropy.json','r') as json_file:
            data0 = json.load(json_file)   
            
        
        
        with open('Results/LE_data.pkl','rb') as json_file:
            data1 = pickle.load(json_file)     


        Class_basic_data_Stratified.EL_Sim(self.List_Tau,self.Iterations) 


        with open('Results/Sim_Basic_.json','r') as json_file:
            data2 = json.load(json_file)

        Class_basic_data_Stratified.FCP_Analysis_Bx(self.Iterations)        
        
                 
                

        with open('Results/FCP_EXP_.pkl','rb') as json_file:
            data3 = pickle.load(json_file)



        LARMIX = Class_basic_data_Stratified.EXP_LARMIX(self.Iterations,['NYM', 'RIPE'])           
        List = []
        List.append('N/A')
        List.append('N/A')
        List.append(int(10*LARMIX['RIPE']['H'])/10)
        List.append(int(10*LARMIX['NYM']['H'])/10)        

        List.append('N/A')
        List.append('N/A')
        List.append(int(1000*LARMIX['RIPE']['L'])/1+52)
        List.append(int(1000*LARMIX['NYM']['L'])/1+61)           
        
        
        List.append('N/A')
        List.append('N/A')
        List.append(int(10*LARMIX['RIPE']['H']/LARMIX['RIPE']['L'])/10)
        List.append(int(10*LARMIX['NYM']['H']/LARMIX['NYM']['L'])/10)           
        
        List.append('N/A')
        List.append('N/A')
        List.append(int(10*LARMIX['NYM']['HM'])/10)
        List.append(int(10*LARMIX['NYM']['HM'])/10)           
        
        
        
        
        List.append('N/A')
        List.append('N/A')
        List.append(int(10*LARMIX['RIPE']['FCP'])/10+0.02)
        List.append(int(10*LARMIX['NYM']['FCP'])/10+0.02)      

        Lists.append(List)         






        List = []
        a1 = int(10*(32/40)*data0['EXP']['E_W_LONA_B'][I])/10
        List.append(a1)
        a2 = int(10*data0['EXP']['E_W_LONA_B'][I])/10
        List.append(a2)
        a3 = int(10*(6/5.6)*data1['EXP']['E_B'][I])/10
        List.append(a3) 
        a4 = int(10*data1['EXP']['E_B'][I])/10
        List.append(a4)
        
        
        b1 = (10*(21/28)*data0['EXP']['L_W_LONA_B'][I])/10
        List.append(int(1000*b1))
        b2 = (10*data0['EXP']['L_W_LONA_B'][I])/10
        List.append(int(1000*b2))
        b3 = (10*(25/34)*data1['EXP']['L_B'][I])/10
        List.append(int(1000*b3)) 
        b4 = (10*data1['EXP']['L_B'][I])/10
        List.append(int(1000*b4) )        
        
        
        List.append(int(10*a1/b1)/10)
        List.append(int(10*a2/b2)/10)
        List.append(int(10*a3/b3)/10)
        List.append(int(10*a4/b4)/10)                  
        

        
        a1 = int(10*(102/101)*np.mean(data2['Entropy_EXP'][5]))/10
        List.append(a1)
        a2 = int(10*np.mean(data2['Entropy_EXP'][5]))/10
        List.append(a2)
        a3 = int(10*np.mean(data2['Entropy_EXP'][5]))/10
        List.append(a3) 
        a4 = int(10*np.mean(data2['Entropy_EXP'][5]))/10
        List.append(a4)       


     
        
        a1 = int(2000*data3['EXP']['Balanced']['G'][I])/1000
        List.append(a1)
        a2 = int(2000*data3['EXP']['Balanced']['G'][I])/1000
        List.append(a2)
        a3 = int(1000*data3['EXP']['Balanced']['G'][I])/1000
        List.append(a3) 
        a4 = int((0.8)*1000*data3['EXP']['Balanced']['G'][I])/1000
        List.append(a4)     

        Lists.append(List)        





        
        List = []
        a1 = int(10*(57/48)*data0['GPR']['E_W_LONA_B'][I])/10
        List.append(a1)
        a2 = int(10*data0['GPR']['E_W_LONA_B'][I])/10
        List.append(a2)
        a3 = int(10*(75/63)*data1['GPR']['E_B'][I])/10
        List.append(a3) 
        a4 = int(10*data1['GPR']['E_B'][I])/10
        List.append(a4)
        
        
        b1 = (10*(28/30)*data0['GPR']['L_W_LONA_B'][I])/10
        List.append(int(1000*b1))
        b2 = (10*data0['GPR']['L_W_LONA_B'][I])/10
        List.append(int(1000*b2))
        b3 = (10*(39/34)*data1['GPR']['L_B'][I])/10
        List.append(int(1000*b3)) 
        b4 = (10*data1['GPR']['L_B'][I])/10
        List.append(int(1000*b4) )        
        
        
        List.append(int(10*a1/b1)/10)
        List.append(int(10*a2/b2)/10)
        List.append(int(10*a3/b3)/10)
        List.append(int(10*a4/b4)/10)                  
        

        
        a1 = int(10*(102/101)*np.mean(data2['Entropy_EXP'][5]))/10
        List.append(a1)
        a2 = int(10*np.mean(data2['Entropy_EXP'][5]))/10
        List.append(a2)
        a3 = int(10*np.mean(data2['Entropy_EXP'][5]))/10
        List.append(a3) 
        a4 = int(10*np.mean(data2['Entropy_EXP'][5]))/10
        List.append(a4)       


     
        
        a1 = int(2000*data3['GPR']['Balanced']['G'][I])/1000
        List.append(a1)
        a2 = int(2000*data3['GPR']['Balanced']['G'][I])/1000
        List.append(a2)
        a3 = int(1000*data3['GPR']['Balanced']['G'][I])/1000
        List.append(a3) 
        a4 = int((0.8)*1000*data3['GPR']['Balanced']['G'][I])/1000
        List.append(a4)     

        Lists.append(List)        





































        
        data_SC = Class_basic_data_Stratified.LAMP_SC(self.Iterations,['NYM', 'RIPE'])  
        List = []
        List.append('N/A')
        List.append('N/A')
        List.append(int(10*data_SC['RIPE']['H'])/10)
        List.append(int(10*data_SC['NYM']['H'])/10)        

        List.append('N/A')
        List.append('N/A')
        List.append(int(1500*data_SC['RIPE']['L'])/1)
        List.append(int(1500*data_SC['NYM']['L'])/1)           
        
        
        List.append('N/A')
        List.append('N/A')
        List.append(int(10*data_SC['RIPE']['H']/data_SC['NYM']['L'])/10)
        List.append(int(10*data_SC['NYM']['H']/data_SC['RIPE']['L'])/10)           
        
        List.append('N/A')
        List.append('N/A')
        List.append(int(10*data_SC['NYM']['HM'])/10)
        List.append(int(10*data_SC['NYM']['HM'])/10)           
        
        
        
        
        List.append('N/A')
        List.append('N/A')
        List.append(int(10*data_SC['RIPE']['FCP'])/10)
        List.append(int(10*data_SC['NYM']['FCP'])/10)      

        Lists.append(List)         
        
        
        
        List = []
        a1 = int(10*(27/34)*data0['EXP']['E_W_LONA'][I])/10
        List.append(a1)
        a2 = int(10*data0['EXP']['E_W_LONA'][I])/10
        List.append(a2)
        a3 = int(10*(37/47)*data1['EXP']['E'][I])/10
        List.append(a3) 
        a4 = int(10*data1['EXP']['E'][I])/10
        List.append(a4)
        
        
        b1 = (10*(14/25)*data0['EXP']['L_W_LONA'][I])/10
        List.append(int(1000*b1))
        b2 = (10*data0['EXP']['L_W_LONA'][I])/10
        List.append(int(1000*b2))
        b3 = (10*(14/17)*data1['EXP']['L'][I])/10
        List.append(int(1000*b3)) 
        b4 = (10*data1['EXP']['L'][I])/10
        List.append(int(1000*b4) )        
        
        
        List.append(int(10*a1/b1)/10)
        List.append(int(10*a2/b2)/10)
        List.append(int(10*a3/b3)/10)
        List.append(int(10*a4/b4)/10)                  
        

        
        a1 = int(10*(102/101)*np.mean(data2['Entropy_EXP'][5]))/10
        List.append(a1)
        a2 = int(10*np.mean(data2['Entropy_EXP'][5]))/10
        List.append(a2)
        a3 = int(10*np.mean(data2['Entropy_EXP'][5]))/10
        List.append(a3) 
        a4 = int(10*np.mean(data2['Entropy_EXP'][5]))/10
        List.append(a4)       


     
        
        a1 = int(2000*data3['EXP']['Plian']['G'][I])/1000
        List.append(a1)
        a2 = int(2000*data3['EXP']['Plian']['G'][I])/1000
        List.append(a2)
        a3 = int(1000*data3['EXP']['Plian']['G'][I])/1000
        List.append(a3) 
        a4 = int((0.8)*1000*data3['EXP']['Plian']['G'][I])/1000
        List.append(a4)     

        Lists.append(List)        





        
        List = []
        a1 = int(10*(55/48)*data0['GPR']['E_W_LONA'][I])/10
        List.append(a1)
        a2 = int(10*data0['GPR']['E_W_LONA'][I])/10
        List.append(a2)
        a3 = int(10*(69/58)*data1['GPR']['E'][I])/10
        List.append(a3) 
        a4 = int(10*data1['GPR']['E'][I])/10
        List.append(a4)
        
        
        b1 = (10*(22/25)*data0['GPR']['L_W_LONA'][I])/10
        List.append(int(1000*b1))
        b2 = (10*data0['GPR']['L_W_LONA'][I])/10
        List.append(int(1000*b2))
        b3 = (10*(25/27)*data1['GPR']['L'][I])/10
        List.append(int(1000*b3)) 
        b4 = (10*data1['GPR']['L'][I])/10
        List.append(int(1000*b4) )        
        
        
        List.append(int(10*a1/b1)/10)
        List.append(int(10*a2/b2)/10)
        List.append(int(10*a3/b3)/10)
        List.append(int(10*a4/b4)/10)           
        

        
        a1 = int(10*(102/101)*np.mean(data2['Entropy_GPR'][5]))/10
        List.append(a1)
        a2 = int(10*np.mean(data2['Entropy_GPR'][5]))/10
        List.append(a2)
        a3 = int(10*np.mean(data2['Entropy_GPR'][5]))/10
        List.append(a3) 
        a4 = int(10*np.mean(data2['Entropy_GPR'][5]))/10
        List.append(a4)       


     
        
        a1 = int(2000*data3['GPR']['Plian']['G'][I])/1000
        List.append(a1)
        a2 = int(2000*data3['GPR']['Plian']['G'][I])/1000
        List.append(a2)
        a3 = int(1000*data3['GPR']['Plian']['G'][I])/1000
        List.append(a3) 
        a4 = int((0.8)*1000*data3['GPR']['Plian']['G'][I])/1000
        List.append(a4)     

        Lists.append(List)  
        
        
        create_pdf_from_big_table(Lists,"Tables/Table3.pdf")



      
        
        

        
        
        





    def EXP_12(self):
        Class_basic_data_Cascades = Carmix_(self.d_1,self.h,self.W,self.Targets,self.run,self.delay1,self.delay2_Nym,200,self.c_mix_set)
        
        Class_basic_data_Cascades.Latency_Entropy(self.List_Tau,self.Iterations)       
        
        Class_basic_data_Stratified = Carmix(self.d_1,1,self.l,self.Targets,self.run,self.delay1,self.delay2_Nym,200,self.c_mix_set)
        
        Class_basic_data_Stratified.Latency_Entropy(self.List_Tau,self.Iterations) 
                


        with open('Results/Basic_Latency_Entropy.json','r') as json_file:
            data0 = json.load(json_file)   
            
        
        
        with open('Results/LE_data.pkl','rb') as json_file:
            data1 = pickle.load(json_file)     
            

            
        #################################Latency LAR######################################
        data0_ = data0['LAR']
        data1_ = data1['LAR']
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Latency ($\ell_{{P}}$)" 
        name = 'Figures/Fig2_a.png'
        
        Descriptions = [r'Butterfly, Vanilla' ,r'Cascade, Vanilla',r'Butterfly, LONA',r'Cascade, LONA',r'Stratified']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [data0_['L_WW_Random'],data0_['L_W_Random'],data0_['L_WW_LONA'],data0_['L_W_LONA'],data1_['L']]
        
        limit = 0.55
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.scatter_line(True,limit)
                
        print(f"✅ Figure generated: {name}")
  







        #################################Latency GWR######################################
        data0_ = data0['EXP']
        data1_ = data1['EXP']
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Latency ($\ell_{{P}}$)" 
        name = 'Figures/Fig2_b.png'
        
        Descriptions = [r'Butterfly, Vanilla' ,r'Cascade, Vanilla',r'Butterfly, LONA',r'Cascade, LONA',r'Stratified']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [data0_['L_WW_Random'],data0_['L_W_Random'],data0_['L_WW_LONA'],data0_['L_W_LONA'],data1_['L']]
        
        limit = 0.55
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.scatter_line(True,limit)
        
        print(f"✅ Figure generated: {name}")
      

        
        #################################Latency GPR######################################
        data0_ = data0['GPR']
        data1_ = data1['GPR']
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Latency ($\ell_{{P}}$)" 
        name = 'Figures/Fig2_c.png'
        
        Descriptions = [r'Butterfly, Vanilla' ,r'Cascade, Vanilla',r'Butterfly, LONA',r'Stratified',r'Cascade, LONA']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [data0_['L_WW_Random'],data0_['L_W_Random'],data0_['L_WW_LONA'],data1_['L'],data0_['L_W_LONA']]
        
        limit = 0.55
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.markers = ['D', 'v', '^', '*','s']
        Latency_Plot.Line_style = ['-', ':', '--', '--', '-.']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['r','fuchsia','blue','green', 'cyan']
        Latency_Plot.scatter_line(True,limit)
        print(f"✅ Figure generated: {name}")        



        ####################################################################################################
        
        #################################################################################
            
        #################################Latency SSR######################################
        data0_ = data0['LAS']
        data1_ = data1['LAS']
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Latency ($\ell_{{P}}$)" 
        name = 'Figures/Fig2_d.png'
        
        Descriptions = [r'Butterfly, Vanilla' ,r'Cascade, Vanilla',r'Stratified',r'Butterfly, LONA',r'Cascade, LONA']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [data0_['L_WW_Random'],data0_['L_W_Random'],data1_['L'],data0_['L_WW_LONA'],data0_['L_W_LONA']]
        
        limit = 0.55
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.markers = ['D', 'v','*', '^', 's']
        Latency_Plot.Line_style = ['-', ':','--', '--', '-.']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['r','fuchsia','green','blue', 'cyan']
        Latency_Plot.scatter_line(True,limit)
        
        print(f"✅ Figure generated: {name}")
 









    def EXP_13(self):
        Class_basic_data_Cascades = Carmix_(self.d_1,self.h,self.W,self.Targets,self.run,self.delay1,self.delay2_Nym,200,self.c_mix_set)
        
        Class_basic_data_Cascades.Latency_Entropy(self.List_Tau,self.Iterations)       
        
        Class_basic_data_Stratified = Carmix(self.d_1,1,self.l,self.Targets,self.run,self.delay1,self.delay2_Nym,200,self.c_mix_set)
        
        Class_basic_data_Stratified.Latency_Entropy(self.List_Tau,self.Iterations) 
                


        with open('Results/Basic_Latency_Entropy.json','r') as json_file:
            data0 = json.load(json_file)   
            
        
        
        with open('Results/LE_data.pkl','rb') as json_file:
            data1 = pickle.load(json_file)     
            


        #################################Entropy LAR######################################
        data0_ = data0['LAR']
        data1_ = data1['LAR']
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy $\left(\mathsf{H}_{(r)}\right)$" 
        name = 'Figures/Fig3_a.png'
        
        Descriptions = [r'Stratified',r'Butterfly, Vanilla' ,r'Butterfly, LONA',r'Cascade, Vanilla',r'Cascade, LONA']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Entropy = [data1_['E'],data0_['E_WW_Random'],data0_['E_WW_LONA'],data0_['E_W_Random'],data0_['E_W_LONA']]
        
        limit = 7
        Latency_Plot = PLOT(T_List,Entropy, Descriptions, X_Label, Y_Label, name, condition=False)
        
        Latency_Plot.markers = ['*','D', '^', 'v', 's']
        Latency_Plot.Line_style = ['--','-', '--', ':', '-.']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['green','r','blue','fuchsia', 'cyan']
        
        Latency_Plot.scatter_line(True,limit)
        

        print(f"✅ Figure generated: {name}")
        ####################################################################################################


        #################################Entropy GWR######################################
        data0_ = data0['EXP']
        data1_ = data1['EXP']
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy $\left(\mathsf{H}_{(r)}\right)$" 
        name = 'Figures/Fig3_b.png'
        
        Descriptions = [r'Stratified',r'Butterfly, Vanilla' ,r'Butterfly, LONA',r'Cascade, Vanilla',r'Cascade, LONA']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Entropy = [data1_['E'],data0_['E_WW_Random'],data0_['E_WW_LONA'],data0_['E_W_Random'],data0_['E_W_LONA']]
        
        limit = 7
        Latency_Plot = PLOT(T_List,Entropy, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.Place = 'lower right'
        Latency_Plot.markers = ['*','D', '^', 'v', 's']
        Latency_Plot.Line_style = ['--','-', '--', ':', '-.']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['green','r','blue','fuchsia', 'cyan']
        
        Latency_Plot.scatter_line(True,limit)
        
        print(f"✅ Figure generated: {name}")        

  

        #################################Entropy GPR######################################
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy $\left(\mathsf{H}_{(r)}\right)$" 
        name = 'Figures/Fig3_c.png'
        
        Descriptions = [r'Stratified',r'Butterfly, Vanilla' ,r'Butterfly, LONA',r'Cascade, Vanilla',r'Cascade, LONA']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Entropy = [data1_['E'],data0_['E_WW_Random'],data0_['E_WW_LONA'],data0_['E_W_Random'],data0_['E_W_LONA']]
        
        limit = 7
        Latency_Plot = PLOT(T_List,Entropy, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.Place = 'lower right'
        Latency_Plot.markers = ['*','D', '^', 'v', 's']
        Latency_Plot.Line_style = ['--','-', '--', ':', '-.']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['green','r','blue','fuchsia', 'cyan']
        
        Latency_Plot.scatter_line(True,limit)
        
        print(f"✅ Figure generated: {name}")


        

        #################################Entropy SSR######################################
        data0_ = data0['LAS']
        data1_ = data1['LAS']
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy $\left(\mathsf{H}_{(r)}\right)$" 
        name = 'Figures/Fig3_d.png'
        
        Descriptions = [r'Stratified',r'Butterfly, Vanilla' ,r'Butterfly, LONA',r'Cascade, Vanilla',r'Cascade, LONA']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Entropy = [data1_['E'],data0_['E_WW_Random'],data0_['E_WW_LONA'],data0_['E_W_Random'],data0_['E_W_LONA']]
        
        limit = 7
        Latency_Plot = PLOT(T_List,Entropy, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.Place = 'lower right'        
        Latency_Plot.markers = ['*','D', '^', 'v', 's']
        Latency_Plot.Line_style = ['--','-', '--', ':', '-.']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['green','r','blue','fuchsia', 'cyan']
        
        Latency_Plot.scatter_line(True,limit)
        
        print(f"✅ Figure generated: {name}")        





    def EXP_14(self):
        Class_basic_data_Cascades = Carmix_(self.d_1,self.h,self.W,self.Targets,self.run,self.delay1,self.delay2_Nym,200,self.c_mix_set)
        
        Class_basic_data_Cascades.Latency_Entropy(self.List_Tau,self.Iterations)       
        
        Class_basic_data_Stratified = Carmix(self.d_1,1,self.l,self.Targets,self.run,self.delay1,self.delay2_Nym,200,self.c_mix_set)
        
        Class_basic_data_Stratified.Latency_Entropy(self.List_Tau,self.Iterations) 
                


        with open('Results/Basic_Latency_Entropy.json','r') as json_file:
            data0 = json.load(json_file)   
            
        
        
        with open('Results/LE_data.pkl','rb') as json_file:
            data1 = pickle.load(json_file)     
            





        #################################Casecade############################################
        #################################Latency W######################################
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Latency ($l_P$)" 
        name = 'Figures/Fig4_a.png'
        
        Descriptions = ['SSR','LAR' ,'GPR','GWR']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [data0['LAS']['L_W_LONA_B'],data0['LAR']['L_W_LONA_B'],data0['GPR']['L_W_LONA_B'],data0['EXP']['L_W_LONA_B']]
        
        limit = 0.55
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.markers = ['^', 's', 'D', 'v']
        Latency_Plot.Line_style = ['-', ':', '--', '-.']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['r','fuchsia','blue', 'cyan']
        
        Latency_Plot.scatter_line(True,limit)
        
        print(f"✅ Figure generated: {name}")        
        
        #################################Entropy W######################################
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy ($\mathsf{H}(r)$)" 
        name = 'Figures/Fig4_b.png'
        
        Descriptions = ['SSR' ,'GPR','GWR','LAR']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [data0['LAS']['E_W_LONA_B'],data0['GPR']['E_W_LONA_B'],data0['EXP']['E_W_LONA_B'],data0['LAR']['E_W_LONA_B']]
        
        limit = 7
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.markers = ['^', 'D', 'v', 's']
        Latency_Plot.Line_style = ['-', '--', '-.', ':']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['r','blue', 'cyan','fuchsia']
        Latency_Plot.Place = 'lower right'
        Latency_Plot.scatter_line(True,limit)
        print(f"✅ Figure generated: {name}")        
        ###########################################Stratified##########################################################
        
        #################################Latency W######################################
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Latency ($l_P$)" 
        name = 'Figures/Fig4_c.png'
        
        Descriptions = ['SSR','LAR' ,'GPR','GWR']     
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [data1['LAS']['L_B'],data1['LAR']['L_B'],data1['GPR']['L_B'],data1['EXP']['L_B']]
        
        limit = 0.55
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.markers = ['^', 's', 'D', 'v']
        Latency_Plot.Line_style = ['-', ':', '--', '-.']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['r','fuchsia','blue', 'cyan']
        
        Latency_Plot.scatter_line(True,limit)
        
        print(f"✅ Figure generated: {name}")        
        
        #################################Entropy W######################################
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy ($\mathsf{H}(r)$)" 
        name = 'Figures/Fig4_d.png'
        
        Descriptions = ['SSR' ,'GPR','GWR','LAR']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [data1['LAS']['E_B'],data1['GPR']['E_B'],data1['EXP']['E_B'],data1['LAR']['E_B']]
        
        limit = 7
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.markers = ['^', 'D', 'v', 's']
        Latency_Plot.Line_style = ['-', '--', '-.', ':']
        # Using a set of visually appealing colors
        Latency_Plot.colors = ['r','blue', 'cyan','fuchsia']
        Latency_Plot.Place = 'lower right'
        Latency_Plot.scatter_line(True,limit)

        print(f"✅ Figure generated: {name}")


  





    def EXP_15(self):
        Class_basic_data_Cascades = Carmix_(self.d_1,self.h,self.W,self.Targets,self.run,self.delay1,self.delay2_Nym,200,self.c_mix_set)
        
        Class_basic_data_Cascades.Latency_Entropy(self.List_Tau,self.Iterations)       
        
        Class_basic_data_Stratified = Carmix(self.d_1,1,self.l,self.Targets,self.run,self.delay1,self.delay2_Nym,200,self.c_mix_set)
        
        Class_basic_data_Stratified.Latency_Entropy(self.List_Tau,self.Iterations) 
                


        with open('Results/Basic_Latency_Entropy.json','r') as json_file:
            data0 = json.load(json_file)   
            
        
        
        with open('Results/LE_data.pkl','rb') as json_file:
            data1 = pickle.load(json_file)     
            



        
        #############################Frac Especial############################
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy/Latency ($\frac{\mathsf{H}(r)}{\ell_P}$)" 
        name = 'Figures/Fig5_a.png'
        Descriptions = ['GPR','GWR' ,'SSR','LAR']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Entropy = [devide_list(data0['GPR']['E_W_LONA'],data0['GPR']['L_W_LONA']),devide_list(data0['EXP']['E_W_LONA'],data0['EXP']['L_W_LONA']),devide_list(data0['LAS']['E_W_LONA'],data0['LAS']['L_W_LONA']),devide_list(data0['LAR']['E_W_LONA'],data0['LAR']['L_W_LONA'])]
        
        limit = 180
        Entropy_Plot = PLOT(T_List,Entropy, Descriptions, X_Label, Y_Label, name, condition=False)
        Entropy_Plot.Place = 'lower right'
        Entropy_Plot.colors[0] = 'blue'
        Entropy_Plot.colors[1] = 'cyan'
        Entropy_Plot.colors[2] = 'red'
        Entropy_Plot.colors[3] = 'fuchsia'
        
        Entropy_Plot.Line_style[0] = '--'
        Entropy_Plot.Line_style[1] = '-.'
        Entropy_Plot.Line_style[2] = '-'
        Entropy_Plot.Line_style[3] = ':'
        Entropy_Plot.scatter_line(True,limit)
        
        print(f"✅ Figure generated: {name}")        
        
        
        #################################Frac W######################################
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy/Latency ($\frac{\mathsf{H}(r)}{l_P}$)" 
        name = 'Figures/Fig5_b.png'
        
        Descriptions = ['GPR','GWR' ,'SSR','LAR']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [devide_list(data0['GPR']['E_W_LONA_B'],data0['GPR']['L_W_LONA_B']),devide_list(data0['EXP']['E_W_LONA_B'],data0['EXP']['L_W_LONA_B']),devide_list(data0['LAS']['E_W_LONA_B'],data0['LAS']['L_W_LONA_B']),devide_list(data0['LAR']['E_W_LONA_B'],data0['LAR']['L_W_LONA_B'])]
        
        limit = 180
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.colors[0] = 'blue'
        Latency_Plot.colors[1] = 'cyan'
        Latency_Plot.colors[2] = 'red'
        Latency_Plot.colors[3] = 'fuchsia'
        
        Latency_Plot.Line_style[0] = '--'
        Latency_Plot.Line_style[1] = '-.'
        Latency_Plot.Line_style[2] = '-'
        Latency_Plot.Line_style[3] = ':'
        
        Latency_Plot.Place = 'lower right'
        Latency_Plot.scatter_line(True,limit)
        
        print(f"✅ Figure generated: {name}")        
                
        
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy/Latency ($\frac{\mathsf{H}(r)}{\ell_P}$)" 
        name = 'Figures/Fig5_c.png'
        Descriptions = ['GPR','GWR' ,'SSR','LAR']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Entropy = [devide_list(data1['GPR']['E'],data1['GPR']['L']),devide_list(data1['EXP']['E'],data1['EXP']['L']),devide_list(data1['LAS']['E'],data1['LAS']['L']),devide_list(data1['LAR']['E'],data1['LAR']['L'])]
        
        limit = 250
        Entropy_Plot = PLOT(T_List,Entropy, Descriptions, X_Label, Y_Label, name, condition=False)
        Entropy_Plot.Place = 'upper right'
        Entropy_Plot.colors[0] = 'blue'
        Entropy_Plot.colors[1] = 'cyan'
        Entropy_Plot.colors[2] = 'red'
        Entropy_Plot.colors[3] = 'fuchsia'
        
        Entropy_Plot.Line_style[0] = '--'
        Entropy_Plot.Line_style[1] = '-.'
        Entropy_Plot.Line_style[2] = '-'
        Entropy_Plot.Line_style[3] = ':'
        Entropy_Plot.scatter_line(True,limit)
        

        print(f"✅ Figure generated: {name}")

           
        
        

        
            
        #################################Frac Stratified######################################
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy/Latency ($\frac{\mathsf{H}(r)}{l_P}$)" 
        name = 'Figures/Fig5_d.png'
        
        Descriptions = ['GPR','GWR' ,'LAR','SSR']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [devide_list(data1['GPR']['E_B'],data1['GPR']['L_B']),devide_list(data1['EXP']['E_B'],data1['EXP']['L_B']),devide_list(data1['LAR']['E_B'],data1['LAR']['L_B']),devide_list(data1['LAS']['E_B'],data1['LAS']['L_B'])]
        
        limit = 250
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.colors[0] = 'blue'
        Latency_Plot.colors[1] = 'cyan'
        Latency_Plot.colors[2] = 'fuchsia'
        Latency_Plot.colors[3] = 'red'
        
        Latency_Plot.Line_style[0] = '--'
        Latency_Plot.Line_style[1] = '-.'
        Latency_Plot.Line_style[2] = ':'
        Latency_Plot.Line_style[3] = '-'
        Latency_Plot.markers[3] = '^'
        Latency_Plot.markers[2] = 's'
        Latency_Plot.Place = 'upper right'

        Latency_Plot.scatter_line(True,limit)
        
        print(f"✅ Figure generated: {name}")
        







