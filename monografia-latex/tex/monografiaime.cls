\ProvidesClass{monografiaime}

\LoadClass{book}

\ExecuteOptions{12pt,twoside,a4paper}

\DeclareOption{quali}{\def\isquali{1}}

\ProcessOptions*

% ---------------------------------------------------------------------------- %
% Pacotes

% Ajuda a hifenização de palavras com acento.
\RequirePackage[T1]{fontenc}

% Coloca os conteúdos auto-gerados, como "Figura", em português.
\RequirePackage[brazil]{babel}

% Aceita caracteres UTF-8 no arquivo fonte.
\RequirePackage[latin1,utf8]{inputenc}

% Usamos arquivos pdf/png como figuras.
\RequirePackage[pdftex]{graphicx}
\RequirePackage{wrapfig}

% Espaçamento flexível.
\RequirePackage{setspace}

% Indentação do primeiro parágrafo.
\RequirePackage{indentfirst}

% Índice remissivo.
\RequirePackage{makeidx}

% Acrescentamos a bibliografia/indice/conteudo no Table of Contents.
\RequirePackage[nottoc]{tocbibind}

% Usa o Adobe Courier no lugar de Computer Modern Typewriter.
\RequirePackage{courier}

% Fontes realmente escaláveis.
\RequirePackage{type1cm}

% Para formatar código-fonte (ex. em Java).
\RequirePackage{listings}
% Para enquadrar código-fonte
\lstset{
numbers=left,
numberstyle=\small,
numbersep=8pt,
frame=single,
framexleftmargin=15pt}
}

% Habilita o uso de matrix.
\RequirePackage{amsmath}
\RequirePackage{mathtools}
\RequirePackage{amssymb}

% Permite alinhar verticalmente o texto dentro das células das tabelas.
\usepackage{array}

% Habilita ambiente de algoritmos.
\RequirePackage[linesnumbered,portuguese]{algorithm2e}

\RequirePackage{titletoc}
\RequirePackage[fixlanguage]{babelbib}
\RequirePackage[font=small,format=plain,labelfont=bf,up,textfont=up]{caption}
\RequirePackage{subcaption}
\RequirePackage[usenames,svgnames,dvipsnames]{xcolor}

% Margens.
\RequirePackage[%
    a4paper, top=3cm, bottom=2.25cm, inner=3cm, outer=2cm%
]{geometry}

% Links coloridos.
\RequirePackage[pdftex, plainpages=false, pdfpagelabels, pagebackref,%
                colorlinks=true]{hyperref}
\AtBeginDocument{
    \hypersetup{
        bookmarksopen=true,    % show bookmarks bar?
        pdftoolbar=true,       % show Acrobat’s toolbar?
        pdfmenubar=true,       % show Acrobat’s menu?
        pdffitwindow=false,    % window fit to page when opened
        pdftitle={\@title},
        pdfauthor={\@author},
        pdfcreator={pdflatex}, % creator of the document
        pdfnewwindow=true,     % links in new window
        %colorlinks=true,       % false: boxed links; true: colored links
        linkcolor=NavyBlue,    % color of internal links
        citecolor=DarkGreen,   % color of links to bibliography
        filecolor=green,       % color of file links
        urlcolor=DarkRed       % color of external links
    }
}

% Soluciona o problema com o hyperref e capitulos.
\RequirePackage[all]{hypcap}

% Citação bibliográfica alpha (alpha-ime.bst).
\RequirePackage[square,sort,nonamebreak,comma]{natbib}

% Para citações
\RequirePackage{csquotes}

\RequirePackage{ifthen}
\RequirePackage{etoolbox}
\RequirePackage{color}
\RequirePackage{calc}
\RequirePackage{picture}
\RequirePackage{lipsum}
\RequirePackage{framed}
\RequirePackage{booktabs}
\RequirePackage{dcolumn}
\RequirePackage{tabularx}
\RequirePackage{enumitem}

\fontsize{60}{62}\usefont{OT1}{cmr}{m}{n}{\selectfont}

% ---------------------------------------------------------------------------- %
% Permite usar os comandos \textvar{texto} para escrever com a mesma fonte das
% variáveis do ambiente de algorítmos.
\DeclareTextFontCommand{\textvar}{\fontfamily{cmss}\selectfont}

% ---------------------------------------------------------------------------- %
% Permite usar os comandos \textfunc{texto} para escrever com a mesma fonte das
% funções do ambiente de algorítmos.
\DeclareTextFontCommand{\textfunc}{\fontfamily{qcr}\selectfont}

% ---------------------------------------------------------------------------- %
% Cabeçalhos similares ao TAOCP de Donald E. Knuth.
\RequirePackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\chaptermark}[1]{\markboth{\MakeUppercase{#1}}{}}
\renewcommand{\sectionmark}[1]{\markright{\MakeUppercase{#1}}{}}
\renewcommand{\headrulewidth}{0pt}

% ---------------------------------------------------------------------------- %

% Caminho das figuras (recomendável).
\graphicspath{{./images/}}

% Arruma o espaço: id est (i.e.) e exempli gratia (e.g.).
\frenchspacing

% URL com o mesmo estilo do texto e não mono-spaced.
\urlstyle{same}

% Para o índice remissivo.
\makeindex

% Para não permitir espaços extra no texto.
\raggedbottom

\fontsize{60}{62}\usefont{OT1}{cmr}{m}{n}{\selectfont}
\cleardoublepage
\normalsize
\inputencoding{utf8}

\csdef{monoime@nomealuno}{Nome do Aluno (\textbackslash nome)}
\newcommand\nome[1]{
  \csdef{monoime@nomealuno}{#1}
  \author{#1}
}
\csdef{monoime@titulotrabalho}{Titulo do trabalho (\textbackslash titulo)}
\newcommand\titulo[1]{
  \csdef{monoime@titulotrabalho}{#1}
  \title{#1}
}
\csdef{monoime@objetivotrabalho}{obtenção do título}
\csdef{monoime@tipotrabalho}{Tipo Trabalho}
\csdef{monoime@tipotrabalhoapresentado}{%
    \csuse{monoime@tipotrabalho} apresentado%
}
\csdef{monoime@nivel}{%
    Nivel (\textbackslash mestrado ou \textbackslash doutorado)%
}
\csdef{monoime@tituloalvo}{%
    Titulo alvo (\textbackslash mestrado ou \textbackslash doutorado)%
}

\csdef{monoime@programa}{Nome do Programa}
\newcommand\programa[1]{
  \csdef{monoime@programa}{#1}
}

\newcommand\mestrado[1]{
    \csdef{monoime@nivel}{Bacharelado}
    \csdef{monoime@tituloalvo}{Bacharel}
    \csdef{monoime@tipotrabalho}{Monografia}
    \csdef{monoime@tipotrabalhoapresentado}{%
        \csuse{monoime@tipotrabalho} apresentada%
    }
    \programa{#1}
}
\newcommand\doutorado[1]{
    \csdef{monoime@nivel}{Doutorado}
    \csdef{monoime@tituloalvo}{Doutor}
    \csdef{monoime@tipotrabalho}{Tese}
    \csdef{monoime@tipotrabalhoapresentado}{%
        \csuse{monoime@tipotrabalho} apresentada%
    }
    \programa{#1}
}
\csdef{monoime@alvotrabalho}{\csuse{monoime@tituloalvo}}
\csdef{monoime@orientador}{%
    Nome do orientador (\textbackslash orientador\{Nome\})%
}
\csdef{monoime@institutoorientador}{IME-USP}
\newcommand\orientador[2]{
    \csdef{monoime@orientador}{#1}
    \csdef{monoime@institutoorientador}{#2}
}
\newcommand\coorientador[2]{
    \def\comcoorientador{1}
    \csdef{monoime@coorientador}{#1}
    \csdef{monoime@institutocoorientador}{#2}
}
\def\orientadorcominstituto{%
    \csuse{monoime@orientador} - \csuse{monoime@institutoorientador}%
}
\def\apoiofinanceirobool{0}
\newcommand\apoiofinanceiro[1]{
    \csdef{monoime@apoiofinanceiro}{#1}
    \def\apoiofinanceirobool{1}
}
\csdef{monoime@datadeposito}{mes de ano (\textbackslash datadeposito)}
\newcommand\datadeposito[1]{
    \csdef{monoime@datadeposito}{#1}
}

\def\defesarealizada{0}
\csdef{monoime@datadefesa}{00/00/0000}
\csdef{monoime@prof1}{Prof. Nome 1 - IME-USP}
\csdef{monoime@prof2}{Prof. Nome 2 - UNICAMP}
\csdef{monoime@prof3}{Prof. Nome 3 - IME-USP}
\newcommand\dadosdefesa[4]{%
    \def\defesarealizada{1}
    \csdef{monoime@datadefesa}{#1}
    \csdef{monoime@prof1}{#2}
    \csdef{monoime@prof2}{#3}
    \csdef{monoime@prof3}{#4}
}

\newcommand\newblankpage{\newpage\null\thispagestyle{empty}\newpage}

% ---------------------------------------------------------------------------- %
% CAPA
% Nota: O título para as dissertações/teses do IME-USP devem caber em um
% orifício de 10,7cm de largura x 6,0cm de altura que há na capa fornecida pela
% SPG.

\newsavebox{\monoime@visorbox}
\newsavebox{\monoime@visorinnerbox}
\ifx\debugcapaime\undefined
    \newenvironment{visorcapa}%
    {\begin{minipage}[t][5.7cm][c]{9.7cm}
     \centering
     \begin{minipage}[t][5.4cm][t]{9.4cm}}%
    {\end{minipage}
     \end{minipage}}
\else
    \newenvironment{visorcapa}%
    {\begin{lrbox}{\monoime@visorbox}
     \begin{minipage}[t][5.7cm][c]{9.7cm}
     \centering
     \begin{lrbox}{\monoime@visorinnerbox}
     \centering
     \begin{minipage}[t][5.4cm][t]{9.4cm}}
    {\end{minipage}
     \end{lrbox}\fbox{\usebox{\monoime@visorinnerbox}}
     \end{minipage}
     \end{lrbox}\fbox{\usebox{\monoime@visorbox}}}
\fi

\newcommand\mostrarinstitutoorientadorcapa{
    \def\mostrarinstorient{1}
}
\newcommand\mostrarinstitutocoorientadorcapa{
    \def\mostrarinstcoorient{1}
}

\newcommand\capas{
    \newgeometry{margin=3cm}
    \ifx\isquali\undefined
    \else
        \csdef{monoime@tipotrabalhoapresentado}{Texto apresentado}
        \csdef{monoime@objetivotrabalho}{exame de qualificação}
        \csdef{monoime@alvotrabalho}{\csuse{monoime@nivel}}
    \fi

    \onehalfspacing % espaçamento
    \thispagestyle{empty}

    \begin{center}
        \vspace*{1.0cm}
        \begin{visorcapa}
            \centering
            \vspace{0.75cm}
            \textbf{\Large{\csuse{monoime@titulotrabalho}}}

            \vfill
            \Large{\csuse{monoime@nomealuno}}
            \vspace{1.0cm}
        \end{visorcapa}

        \vskip 2cm
      %  \Large{%
        %    \csuse{monoime@tipotrabalhoapresentado}\\[-0.5em]
        %    ao\\[-0.5em]
          %  Instituto de Matemática e Estatística\\[-0.5em]
          %  da\\[-0.5em]
        %    Universidade de São Paulo\\[-0.5em]
        %    para\\[-0.5em]
        %    \csuse{monoime@objetivotrabalho}\\[-0.5em]
        %    de\\[-0.5em]
        %    \csuse{monoime@alvotrabalho} em \csuse{monoime@programa}
      %  }

        \vskip 2cm
        \large{%
            \begin{flushright}
                \emph{Programa:}\\
                \vspace{-0.3em}
                \csuse{monoime@nivel} em \csuse{monoime@programa}

                \vspace{1em}
                \emph{Orientador:}\\
                \vspace{-0.3em}
                \csuse{monoime@orientador}%
                    \ifx\mostrarinstorient\undefined
                    \else - \csuse{monoime@institutoorientador}
                    \fi \\
                \ifx\comcoorientador\undefined
                \else
                \emph{Coorientador:}\\
                \vspace{-0.3em}
                \csuse{monoime@coorientador}%
                    \ifx\mostrarinstcoorient\undefined
                    \else - \csuse{monoime@institutocoorientador}
                    \fi
                \fi
            \end{flushright}
        }

        \vskip 2cm
        \if\apoiofinanceirobool1
            \normalsize{%
                Durante o desenvolvimento deste trabalho o autor recebeu auxílio
                financeiro d\csuse{monoime@apoiofinanceiro}%
            }
        \else
            ~
        \fi

        \vskip 1.2cm
        \normalsize{São Paulo, \csuse{monoime@datadeposito}}
    \end{center}

    \newblankpage

    \ifx\isquali\undefined
        \if\defesarealizada1
            \folharostofinal
        \else
            \folharostoapresentacao
        \fi
        \newpage
    \fi

    \restoregeometry
    \newblankpage
}

\newcommand\folharostoapresentacao{
    \newgeometry{margin=3cm}
    \onehalfspacing  % espaçamento
    \thispagestyle{empty}

    \begin{center}
        \vspace*{1.0cm}
        \begin{visorcapa}
            \centering
            \vspace{0.75cm}
            \textbf{\Large{\csuse{monoime@titulotrabalho}}}
        \end{visorcapa}
    \end{center}

    \vskip 2cm

    \begin{flushright}
        Esta é a versão original da \MakeLowercase{\csuse{monoime@tipotrabalho}}
        elaborada pelo\\
        aluno \csuse{monoime@nomealuno}\\
    \end{flushright}
    \restoregeometry
}

\newcommand\folharostofinal{
    \newgeometry{margin=3cm}
    \onehalfspacing  % espaçamento
    \thispagestyle{empty}

    \begin{center}
        \vspace*{1.0cm}
        \begin{visorcapa}
            \centering
            \vspace{0.75cm}
            \textbf{\Large{\csuse{monoime@titulotrabalho}}}
        \end{visorcapa}
    \end{center}

    \vskip 2cm

    \begin{flushright}
        Esta versão da \MakeLowercase{\csuse{monoime@tipotrabalho}} contém as
        correções e alterações sugeridas\\
        pela Comissão Julgadora durante a defesa da versão original do
        trabalho,\\
        realizada em \csuse{monoime@datadefesa}. Uma cópia da versão original
        está disponível no\\
        Instituto de Matemática e Estatística da Universidade de São Paulo.
    \end{flushright}
    \vskip 5cm

    \centering
    \begin{minipage}[t][5cm][t]{14cm}
        \noindent Comissão Julgadora:

        \begin{itemize}
            \item \csuse{monoime@prof1}
            \item \csuse{monoime@prof2}
            \item \csuse{monoime@prof3}
        \end{itemize}
    \end{minipage}
    \restoregeometry
}

\newcommand\formatacaopaginaspreliminares{
    \frontmatter
    \fancyhead[RO]{{\footnotesize\rightmark}\hspace{2em}\thepage}
    \setcounter{tocdepth}{2}
    \fancyhead[LE]{\thepage\hspace{2em}\footnotesize{\leftmark}}
    \fancyhead[RE,LO]{}
}

\newcommand\formatacaocorpotrabalho[1][\onehalfspacing]{
    #1
    \mainmatter
    \fancyhead[RE,LO]{\thesection}
}

\newcommand\formatacaoapendices{
    \renewcommand{\chaptermark}[1]{%
        \markboth{%
            \MakeUppercase{\appendixname\ \thechapter}%
        }{\MakeUppercase{##1}}%
    }
    \fancyhead[RE,LO]{}
}
