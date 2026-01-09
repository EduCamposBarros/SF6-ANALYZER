def generate_insights(frame_data_result):
    """
    Recebe o resultado de `calculate_frame_data` (dicionário com 'windows' e 'summary')
    e gera uma lista de insights e recomendações acionáveis.
    """

    insights = []

    if not frame_data_result:
        insights.append("Nenhuma troca válida detectada para análise de frame data.")
        return insights

    windows = frame_data_result.get("windows", [])
    summary = frame_data_result.get("summary", {})

    # Estatísticas básicas
    plus = summary.get("plus_on_block", 0)
    minus = summary.get("minus_on_block", 0)
    avg = summary.get("avg_on_block", 0)

    if minus > plus and minus > 3:
        insights.append("Frequente uso de golpes negativamente recompensados no bloqueio; prefira golpes mais seguros ou cancelações.")

    if avg < -3:
        insights.append("Vantagem média após bloqueios é negativa; trabalhe posicionamento e timings para reduzir gap frames.")

    """
    Gera recomendações práticas a partir do resultado de `calculate_frame_data`.

    A função converte métricas numéricas e listas (p.ex. `punishable_jumps`, `whiff_punishes`)
    em mensagens legíveis que indicam pontos fracos e exercícios a praticar.
    """

    # Whiff punish opportunities
    whiffs = summary.get("whiff_punishes", [])
    whiff_count = sum(1 for w in whiffs if w.get("punishable"))
    if whiff_count:
        insights.append(f"Detectados {whiff_count} whiff punish possíveis — pratique punir recovery de ataques perdidos.")

    # Drive impacts
    if summary.get("drive_impacts"):
        insights.append("Foram detectados drive impacts; pratique reagir com defesa ativa (reversals / escape).")

    # Recomendações gerais
    if plus < minus:
        insights.append("Sugestão: use normals mais rápidos quando pressionado e evite moves com longos recovery frames.")

    if not insights:
        insights.append("Nenhuma fraqueza óbvia detectada — foque em polir execução e condições de pressão.")

    return insights
