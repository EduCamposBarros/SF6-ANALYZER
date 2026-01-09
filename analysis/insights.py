def generate_insights(frame_data):
    insights = []

    # Se não houver dados, retorna insight informativo
    if not frame_data:
        insights.append("Nenhuma troca válida detectada para análise de frame data.")
        return insights

    negatives = [f for f in frame_data if f["on_block_adv"] < -5]
    if len(negatives) > 3:
        insights.append("Uso frequente de golpes puníveis (-5 ou pior).")

    avg = sum(f["on_block_adv"] for f in frame_data) / len(frame_data)
    if avg < -3:
        insights.append("Pressão média muito negativa após bloqueios.")

    return insights
